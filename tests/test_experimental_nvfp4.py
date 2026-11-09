import pytest
import torch

from ripple_particle_sliders.experimental_nvfp4 import (
    PackedNVFP4Linear, activation_qdq, decode_rows)


def tensors():
    generator = torch.Generator().manual_seed(81)
    packed = torch.randint(0, 256, (7, 16), dtype=torch.uint8, generator=generator)
    scales = torch.tensor([[1., 2.]] * 7).to(torch.float8_e4m3fn)
    return packed, scales, torch.tensor(.125), torch.tensor(.01)


def test_nibble_order_sign_and_both_scales():
    packed = torch.arange(0, 16, dtype=torch.uint8).unsqueeze(0) | 0x70
    scales = torch.tensor([[2., 4.]]).to(torch.float8_e4m3fn)
    decoded = decode_rows(packed, scales, torch.tensor(.25))
    expected = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6.])
    expected[:8] *= .5
    assert torch.equal(decoded[0, ::2], expected)
    assert torch.equal(decoded[0, 1::2], torch.tensor([3.] * 8 + [6.] * 8))


@pytest.mark.parametrize('chunk_rows', [1, 3, 7, 20])
def test_streamed_forward_backward_and_higher_order(chunk_rows):
    p, s, g, a = tensors()
    layer = PackedNVFP4Linear(p, s, g, a, chunk_rows=chunk_rows, activation='w4a16')
    value = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
    weight = decode_rows(p, s, g, dtype=value.dtype)
    assert torch.allclose(layer(value), value @ weight.T, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(layer, (value,), fast_mode=True)
    assert torch.autograd.gradgradcheck(layer, (value,), fast_mode=True)
    assert not list(layer.parameters())


def test_quantized_forward_and_explicit_identity_ste_vjp():
    p, s, g, a = tensors()
    layer = PackedNVFP4Linear(p, s, g, a, chunk_rows=3, activation='w4a4_ste')
    value = torch.randn(2, 32, requires_grad=True)
    output = layer(value)
    expected = activation_qdq(value, a) @ decode_rows(p, s, g).T
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-6)
    grad, = torch.autograd.grad(output.sum(), value)
    assert torch.allclose(grad, decode_rows(p, s, g).sum(0).expand_as(value), atol=1e-6, rtol=1e-6)


def test_only_packed_buffers_saved_for_backward():
    p, s, g, a = tensors()
    saved = []
    layer = PackedNVFP4Linear(p, s, g, a, chunk_rows=3, activation='w4a16')
    value = torch.randn(2, 32, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        layer(value).sum().backward()
    assert len(saved) == 3
    assert all(t is b for t, b in zip(saved, (p, s, g)))


def test_bf16_vjp_accumulates_chunks_in_fp32():
    p, s, g, a = tensors()
    layer = PackedNVFP4Linear(p, s, g, a, chunk_rows=1, activation='w4a16')
    value = torch.randn(1, 32, dtype=torch.bfloat16, requires_grad=True)
    output = layer(value)
    probe = torch.tensor([[1., 100., -.1, 10., -1., -100., .1]], dtype=torch.bfloat16)
    actual, = torch.autograd.grad(output, value, probe)
    weight = decode_rows(p, s, g, dtype=torch.bfloat16)
    expected = (probe.float() @ weight.float()).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_all_zero_activations_and_nonfinite_rejection():
    _, _, _, a = tensors()
    assert torch.equal(activation_qdq(torch.zeros(2, 32), a), torch.zeros(2, 32))
    with pytest.raises(ValueError, match='finite'):
        activation_qdq(torch.full((1, 32), float('nan')), a)


@pytest.mark.parametrize('fault', ['packing', 'scale_dtype', 'scale_shape', 'global_scale', 'input_scale', 'trainable_scale', 'empty_rows'])
def test_invalid_or_trainable_checkpoint_buffers_rejected(fault):
    p, s, g, a = tensors()
    if fault == 'packing': p = p.float()
    if fault == 'scale_dtype': s = s.float()
    if fault == 'scale_shape': s = s[:, :1]
    if fault == 'global_scale': g = torch.tensor(-1.)
    if fault == 'input_scale': a = torch.tensor(0.)
    if fault == 'trainable_scale': g.requires_grad_()
    if fault == 'empty_rows': p, s = p[:0], s[:0]
    with pytest.raises(ValueError):
        PackedNVFP4Linear(p, s, g, a)
