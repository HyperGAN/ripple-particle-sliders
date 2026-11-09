from copy import deepcopy
import json

import pytest
import torch

from ripple_particle_sliders.backend import VENDOR, RippleBackend
from ripple_particle_sliders.compatibility import inspect_config
from ripple_particle_sliders import binding


def test_pinned_nvfp4_config_has_320_bf16_attention_targets():
    report = inspect_config(json.loads((VENDOR / 'config.json').read_text()))
    targets = report['attention_targets']
    assert len(targets) == 320
    assert targets['model.layers.0.self_attn.q_proj'] == [10240, 5120]
    assert targets['model.layers.79.self_attn.o_proj'] == [5120, 10240]
    assert report['native_response_shape'] == [1, 5120]
    assert report['nvfp4_training_qualified'] is False
    assert binding.capabilities({}).response_shape == (1, 5120)
    assert binding.capabilities({}).base_precisions == ('nvfp4',)


def test_requested_nvfp4_refuses_before_any_model_allocation(monkeypatch):
    from particle_sliders.response import open_response
    monkeypatch.setattr(binding, 'RippleBackend', lambda **kw: pytest.fail('model allocated'))
    with pytest.raises(ValueError, match='native NVFP4.*no BF16 fallback'):
        open_response(binding.config(base_precision='nvfp4'))


@pytest.mark.parametrize('change', ['bf16', 'w4a16', 'dynamic', 'group32',
    'quantized_attention', 'bf16_mlp', 'hybrid_layout', 'wrong_width', 'quantized_kv'])
def test_wrong_quantization_or_architecture_rejected(change):
    config = json.loads((VENDOR / 'config.json').read_text())
    quant = config['quantization_config']
    group = quant['config_groups']['group_0']
    if change == 'bf16':
        config.pop('quantization_config')
    elif change == 'w4a16':
        group['input_activations']['num_bits'] = 16
    elif change == 'dynamic':
        group['input_activations']['dynamic'] = True
    elif change == 'group32':
        group['weights']['group_size'] = 32
    elif change == 'quantized_attention':
        quant['ignore'].remove('model.layers.0.self_attn*')
    elif change == 'bf16_mlp':
        quant['ignore'].append('model.layers.79.mlp.down_proj')
    elif change == 'hybrid_layout':
        config['layer_types'][4] = 'sliding_attention'
    elif change == 'wrong_width':
        config['hidden_size'] = 4096
    elif change == 'quantized_kv':
        quant['kv_cache_quant_algo'] = 'FP8'
    with pytest.raises(ValueError):
        inspect_config(config)


def test_exact_clean_base_off_before_and_after_attachment():
    torch.set_num_threads(1)
    backend = RippleBackend(fixture_width=16)
    original = backend.hidden([1, 2, 3]).detach().clone()
    saved = deepcopy(backend.model.state_dict())
    slider = binding.RippleHNDLSlider(backend.model)
    try:
        # Ensure branches are nonzero, so a vacuous all-zero adapter cannot pass.
        with torch.no_grad():
            for adapter in slider.adapters.values():
                adapter.lora_up.weight.normal_(std=.1)
        with slider.scaled(1):
            on = backend.hidden([1, 2, 3])
        assert not torch.equal(original, on)
        with slider.scaled(0):
            assert torch.equal(original, backend.hidden([1, 2, 3]))
        with slider.scaled(-1):
            negative = backend.hidden([1, 2, 3])
        assert not torch.equal(on, negative)
        for key, value in saved.items():
            assert torch.equal(value, backend.model.state_dict()[key])
    finally:
        slider.close()
    assert torch.equal(original, backend.hidden([1, 2, 3]))


def test_pinned_vendor_code_hashes():
    import hashlib
    from pathlib import Path
    receipt = json.loads((Path(__file__).parents[1] / 'PROVENANCE.json').read_text())
    for name, expected in receipt['vendored_files'].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == expected
