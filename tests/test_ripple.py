from copy import deepcopy
import json

import pytest
import torch

from ripple_particle_sliders import binding as product
from ripple_particle_sliders.backend import RippleBackend, text_attention_targets
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import UnsupportedOperation
from particle_sliders.response import open_response

ROWS = [dict(neutral_ids=[1, 2], positive_ids=[1, 3]),
        dict(neutral_ids=[4, 5], positive_ids=[4, 6])]


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def session(**kw):
    return open_response(product.config(fixture=True, fixture_width=16,
                         steps=4, ema_decay=.9, **kw), rows=ROWS)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('width', [16, 32])
def test_official_ripple_hybrid_perinorm_gated_valres_architecture(width):
    host = RippleBackend(fixture_width=width)
    targets = text_attention_targets(host.model)
    assert len(targets) == 20
    cfg = host.model.config
    assert cfg.layer_types == ['sliding_attention'] * 4 + ['full_attention']
    assert cfg.use_valres and cfg.use_gated_attention and cfg.gate_full_precision
    assert cfg.embed_scale == width ** .5
    assert cfg.ffn_clamp_threshold == 20.
    assert cfg.sliding_window == 4
    assert not any('mlp' in n or 'g_proj' in n for n in targets)
    assert host.hidden([1, 2, 3, 4, 5, 6]).shape == (1, 6, width)
    slider = product.RippleHNDLSlider(host.model)
    try:
        before = [(m, m.forward) for m in targets.values()]
        with pytest.raises(ValueError, match='unwrapped'):
            product.RippleHNDLSlider(host.model)
        assert all(m.forward == f for m, f in before)
    finally:
        slider.close()


def test_default_width32_shared_update():
    with open_response(product.config(fixture=True, steps=2), rows=ROWS) as s:
        assert s.evaluate(strength=0).shape == (1, 32)
        initial = s.game.backend.particles.detach().clone()
        s.train()
        assert not torch.equal(initial, s.game.backend.particles)


def test_shared_updates_base_freeze_bank_gradients_and_exact_off():
    with session() as s:
        native = s.game.backend
        base = deepcopy(native.backend.model.state_dict())
        initial = deepcopy(native.slider.state_dict())
        off = s.evaluate(strength=0)
        s.train()
        assert not torch.equal(initial['particles'], native.particles)
        equal(base, native.backend.model.state_dict())
        assert all(p.grad is None for p in native.backend.model.parameters())
        equal(off, s.evaluate(strength=0))
        assert not torch.equal(off, s.evaluate(strength=1))
        s.game.network.zero_grad(set_to_none=True)
        with s.game.network.scaled(1):
            native.generated(s.game.rows[0]).square().sum().backward()
        assert torch.isfinite(native.particles.grad).all()
        assert torch.count_nonzero(native.particles.grad)
        for key, parameter in native.slider.named_parameters():
            assert parameter.grad is not None, key
            assert torch.isfinite(parameter.grad).all(), key
    assert all(m.forward == original for m, original, _ in native.slider._hosts)


def test_split_resume_and_portable_raw_ema(tmp_path):
    with session() as whole:
        whole.train()
        expected = whole.checkpoint()
    with session() as part:
        part.train(stop_after=2)
        part.checkpoint(tmp_path / 'resume.pt')
    with open_response(product.config(fixture=True, fixture_width=16, steps=4,
                                      ema_decay=.9), rows=ROWS,
                       resume=tmp_path / 'resume.pt') as resumed:
        resumed.train()
        equal(expected, resumed.checkpoint())
        for choice in ('raw', 'ema'):
            old = deepcopy(resumed.game.network.state_dict())
            if choice == 'ema':
                resumed.game.network.load_state_dict(resumed.ema.values)
            probes = {v: resumed.evaluate(strength=v) for v in (0., -.5, .5, 1.)}
            resumed.game.network.load_state_dict(old)
            artifact = resumed.export_inference(selection=choice)
            with open_inference(artifact, operation='response') as inference:
                for strength, expected in probes.items():
                    equal(expected, inference.response(ids=[1, 2], strength=strength))


def test_native_refuses_before_model_or_hooks(monkeypatch):
    monkeypatch.setattr(product, 'RippleBackend', lambda **k: pytest.fail('model loaded'))
    with pytest.raises(ValueError, match='not qualified'):
        open_response(product.config(), rows=ROWS)


@pytest.mark.parametrize('key,value', [('fixture', 'true'), ('dummy', False),
    ('fixture_width', 64), ('base_precision', 'bfloat16'), ('lr', True),
    ('penalty_input_domain', 'continuous'), ('pg_source_profile', 'unreviewed')])
def test_unqualified_options_refuse(key, value, monkeypatch):
    monkeypatch.setattr(product, '_native', lambda *a: pytest.fail('model allocated'))
    options = dict(fixture=True)
    options[key] = value
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(product.config(**options), rows=ROWS)


def test_product_source_tampering_refuses(tmp_path, monkeypatch):
    original = json.loads((product.ROOT / 'sources.lock.json').read_text())
    original['product_sources']['backend.py'] = '0' * 64
    (tmp_path / 'sources.lock.json').write_text(json.dumps(original))
    monkeypatch.setattr(product, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='product/host source closure'):
        product.verify_runtime()


def test_routed_projection_vjp_matches_finite_differences():
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        bank = torch.nn.Parameter(torch.randn(128, 4, dtype=torch.float64))
        host = torch.nn.Linear(16, 8, bias=False).double().requires_grad_(False)
        branch = HNDLParticleProjection('test', host, bank, 8, 8, attach=False).double()
        with torch.no_grad():
            branch.lora_up.weight.normal_(std=.1)
        value = torch.randn(1, 1, 16, dtype=torch.float64, requires_grad=True)
        particles = bank.unsqueeze(0).detach().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x, p: branch.graph(value=x, particles=p), (value, particles),
            eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
