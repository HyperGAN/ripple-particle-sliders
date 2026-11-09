from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ripple_particle_sliders import binding
from ripple_particle_sliders.backend import make_fixture
from ripple_particle_sliders.experimental_nvfp4 import PackedNVFP4Linear
from ripple_particle_sliders.streaming import CURRENT, FileTensorStore, StreamingBackend


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_cache(root, width=16):
    torch.manual_seed(178)
    resident = make_fixture(width)
    for layer in resident.model.layers:
        for leaf in ('gate', 'up', 'down'):
            name = leaf + '_proj'
            original = getattr(layer.mlp, name)
            packed = torch.randint(0, 256, (original.out_features, original.in_features//2), dtype=torch.uint8)
            scales = torch.ones(original.out_features, original.in_features//16).to(torch.float8_e4m3fn)
            setattr(layer.mlp, name, PackedNVFP4Linear(packed, scales,
                torch.tensor(.04), torch.tensor(.01), chunk_rows=3))
    manifest = dict(schema='ripple-raw-tensor-cache-v1', source=dict(
        qualification='random packed CPU architecture fixture only', seed=178),
        config=resident.config.to_dict(), tensors={})
    types = {torch.uint8: 'U8', torch.float8_e4m3fn: 'F8_E4M3', torch.float32: 'F32'}
    aliases = {'packed': 'weight', 'scales': 'weight_scale',
               'global_scale': 'weight_scale_2', 'input_scale': 'input_scale'}
    root.mkdir(parents=True, exist_ok=True)
    for name, tensor in resident.state_dict().items():
        if name.startswith('lm_head.'):
            continue
        if '.mlp.' in name:
            parent, leaf = name.rsplit('.', 1)
            name = parent + '.' + aliases[leaf]
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        path = root / (name + '.bin')
        path.write_bytes(payload)
        manifest['tensors'][name] = dict(file=path.name, dtype=types[tensor.dtype],
            shape=list(tensor.shape), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (root/'manifest.json').write_text(json.dumps(manifest, sort_keys=True)+'\n')
    return resident, FileTensorStore(root)


def responses(tmp_path):
    resident, store = fixture_cache(tmp_path/'cache')
    streaming = StreamingBackend(store, chunk_rows=3)
    host = SimpleNamespace(model=resident,
        hidden=lambda ids: resident.model(input_ids=torch.tensor([ids]), use_cache=False,
                                          return_dict=True).last_hidden_state)
    a = binding.RippleHNDLSlider(resident)
    b = binding.RippleHNDLSlider(streaming.model, device='cpu')
    with torch.no_grad():
        for branch in a.adapters.values():
            branch.lora_up.weight.normal_(std=.02)
    b.load_state_dict(a.state_dict())
    return (binding.RippleResponse(host, a, model_identity={'fixture': 'resident'}),
            binding.RippleResponse(streaming, b, model_identity=store.identity), store)


def test_all_five_layers_routing_off_signed_strengths_and_gradients(tmp_path):
    resident, stream, store = responses(tmp_path)
    try:
        before = deepcopy(resident.backend.model.state_dict())
        assert [b.native.self_attn.layer_type for b in stream.backend.model.layers] == [
            'sliding_attention']*4 + ['full_attention']
        for strength in (0., 1., -.5):
            expected = resident._response([1, 2, 3, 4, 5, 6], scale=strength)
            actual = stream._response([1, 2, 3, 4, 5, 6], scale=strength)
            assert torch.equal(actual, expected)
        expected = resident._response([1, 2, 3], scale=1.)
        actual = stream._response([1, 2, 3], scale=1.)
        # Exit strength context and run a distinct OFF observation before backward.
        with torch.no_grad():
            stream._response([1, 2, 3], scale=0.)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for (name, a), (_, b) in zip(resident.slider.named_parameters(), stream.slider.named_parameters()):
            assert a.grad is not None and b.grad is not None, name
            assert torch.isfinite(b.grad).all(), name
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
        assert torch.count_nonzero(stream.particles.grad)
        assert all(p.device.type == 'meta' and not p.requires_grad for p in stream.backend.model.parameters())
        for name, value in before.items():
            assert torch.equal(value, resident.backend.model.state_dict()[name])
        block_bytes = [sum(v['bytes'] for n,v in store.entries.items() if n.startswith(f'model.layers.{i}.')) for i in range(5)]
        assert store.max_active_bytes == max(block_bytes)
        assert store.active_bytes == 0 and CURRENT.get() is None
    finally:
        stream.close()
        resident.close()
        store.close()


def test_streaming_without_shared_checkpoint_refuses_grad_but_allows_no_grad(tmp_path):
    _, store = fixture_cache(tmp_path/'cache')
    backend = StreamingBackend(store)
    with pytest.raises(RuntimeError, match='shared scale-capturing'):
        backend.hidden([1,2])
    with torch.no_grad():
        assert backend.hidden([1,2]).shape == (1,2,16)
    assert store.active_bytes == 0


def test_corrupt_cache_rejected_and_scope_meta_weights_restored(tmp_path):
    resident, stream, store = responses(tmp_path)
    try:
        path = store.root / store.entries['model.layers.2.self_attn.q_proj.weight']['file']
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        with pytest.raises(ValueError, match='hash differs'):
            stream._response([1,2,3], scale=1.)
        assert store.active_bytes == 0 and CURRENT.get() is None
        assert all(p.device.type == 'meta' for p in stream.backend.model.parameters())
    finally:
        stream.close(); resident.close(); store.close()


def test_embedding_row_hash_and_manifest_identity(tmp_path):
    _, store = fixture_cache(tmp_path/'cache')
    expected = store.read('model.embed_tokens.weight', 'cpu')
    ids = torch.tensor([[3, 1, 3, 0]])
    assert torch.equal(store.rows('model.embed_tokens.weight', ids, 'cpu'), expected[ids])
    path = store.root / store.entries['model.embed_tokens.weight']['file']
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match='hash differs'):
        store.rows('model.embed_tokens.weight', ids, 'cpu')


def test_full_model_cache_missing_block_refuses_before_tensor_reads(tmp_path):
    _, store = fixture_cache(tmp_path/'cache')
    manifest = store.manifest
    del manifest['tensors']['model.layers.4.mlp.up_proj.weight']
    (store.root/'manifest.json').write_text(json.dumps(manifest))
    store = FileTensorStore(store.root)
    with pytest.raises(ValueError, match='incomplete block'):
        StreamingBackend(store)
    assert store.load_count == 0


def test_manifest_snapshot_cannot_be_mutated_after_identity_binding(tmp_path):
    _, store = fixture_cache(tmp_path/'cache')
    identity = store.identity
    with pytest.raises(TypeError):
        store.entries['model.norm.weight']['sha256'] = '0'*64
    copy = store.manifest
    copy['source']['seed'] = 0
    assert store.identity == identity


def test_repeat_backward_reloads_blocks_and_preserves_captured_strength(tmp_path):
    resident, stream, store = responses(tmp_path)
    try:
        output = stream._response([1,2,3], scale=.7)
        loss = output.square().sum()
        loss.backward(retain_graph=True)
        expected = {n:p.grad.clone() for n,p in stream.slider.named_parameters()}
        count = store.load_count
        stream.slider.zero_grad(set_to_none=True)
        with torch.no_grad(): stream._response([1,2,3], scale=-.2)
        loss.backward()
        for name, parameter in stream.slider.named_parameters():
            assert torch.equal(parameter.grad, expected[name])
        assert store.load_count > count
        assert store.active_bytes == 0 and CURRENT.get() is None
    finally:
        stream.close(); resident.close(); store.close()


def test_functional_call_failure_restores_meta_and_releases_scope(tmp_path, monkeypatch):
    resident, stream, store = responses(tmp_path)
    try:
        module = stream.backend.model.layers[0].native
        def failure(*args, **kwargs): raise RuntimeError('injected native failure')
        monkeypatch.setattr(module, 'forward', failure)
        with pytest.raises(RuntimeError, match='injected native failure'):
            stream._response([1,2], scale=1.)
        assert all(p.device.type == 'meta' for p in stream.backend.model.parameters())
        assert store.active_bytes == 0 and CURRENT.get() is None
    finally:
        stream.close(); resident.close(); store.close()


def test_released_full_width_game_gate_preserved():
    model = torch.nn.Module()
    model.config = SimpleNamespace(hidden_size=5120)
    with pytest.raises(ValueError, match='width >=170'):
        binding.RippleHNDLSlider(model, device='cpu')
