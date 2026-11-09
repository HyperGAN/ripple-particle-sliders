"""CPU-only bounded test of a real packed Ripple projection, not full model training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import resource
import time

import torch

from ripple_particle_sliders.experimental_nvfp4 import PackedNVFP4Linear, activation_qdq, decode_rows


def official_reference(path):
    """Execute only the reviewed ModelOpt NVFP4 class and conversion definitions.

    Use the original quantize/dequantize code, avoiding installation of its GPU
    package. Three generic shape/amax helpers below handle this divisible-by-16
    tensor; the arithmetic/packing/scaling/rounding belongs to NVIDIA's code.
    """
    source = path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest() != '0340799307f7883a6698019a6e55b9b2666159c5e6bcbc1f6cd3aad944f7eb77':
        raise ValueError('reference must match pinned ModelOpt 0.46.0 source before execution')
    tree = ast.parse(source)
    names = {'e2m1_bounds', 'e2m1_values', '_cast_per_block_scale_to_fp8', 'NVFP4QTensor'}
    selected = [n for n in tree.body if (
        isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names or
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    class ReferenceBase:
        def __init__(self, shape, dtype, data):
            self.metadata, self._quantized_data = dict(shape=shape, dtype=dtype), data
    namespace = dict(torch=torch, BaseQuantizedTensor=ReferenceBase,
        E2M1_MAX=6., E4M3_MAX=448., fp4_compatible=lambda: False,
        reduce_amax=lambda x: x.abs().amax(),
        reduce_block_padding=lambda x, **kw: x,
        reduce_block_amax=lambda x, **kw: x.reshape(*x.shape[:-1], -1, 16).abs().amax(-1))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['NVFP4QTensor']


def read_probe(root):
    receipt = json.loads((root / 'receipt.json').read_text())
    if (receipt['model'], receipt['revision']) != ('ricursiveAI/Ripple-65B-Preview-NVFP4',
            '99eea4477a001a41b86914e41feb4b2ed86105e5'):
        raise ValueError('wrong checkpoint origin')
    tensors = {}
    for name, desc in receipt['tensors'].items():
        path = root / desc['file']
        if path.parent.resolve() != root.resolve() or path.stat().st_size != desc['bytes']:
            raise ValueError('invalid probe file or size')
        data = bytearray(path.read_bytes())
        if hashlib.sha256(data).hexdigest() != desc['sha256']:
            raise ValueError('probe bytes changed')
        dtype = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn, 'F32': torch.float32}[desc['dtype']]
        tensors[name.rsplit('.', 1)[1]] = torch.frombuffer(data, dtype=dtype).reshape(desc['shape'])
    return tensors, receipt


def digest(tensor):
    return hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--modelopt-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(73)
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(meminfo['MemAvailable'].split()[0]) * 1024
    if available < (47 + 2) * 1024**3:
        raise RuntimeError('CPU probe lacks 2 GiB window after preserving 47 GiB peer/global reserve')
    tensors, origin = read_probe(args.data)
    p, s, g, a = [tensors[n] for n in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')]
    before = {n: digest(t) for n, t in tensors.items()}
    reference = official_reference(args.modelopt_source)
    def reference_weight(first, last, dtype):
        q = reference((last-first, p.shape[1]*2), dtype, p[first:last])
        return q.dequantize(dtype=dtype, scale=s[first:last], double_scale=g, block_sizes={-1: 16})
    checks = {}
    # Unchanged NVIDIA decoder versus independent implementation, at beginning,
    # middle and end, and at both model BF16 and high precision probe readouts.
    for dtype in (torch.float32, torch.bfloat16, torch.float64):
        for first in (0, p.shape[0]//2, p.shape[0]-8):
            actual = decode_rows(p[first:first+8], s[first:first+8], g, dtype=dtype)
            expected = reference_weight(first, first+8, dtype)
            assert torch.equal(actual, expected), (dtype, first)
    checks['decoder_exact_vs_modelopt_0_46_0'] = True
    x = torch.randn(1, p.shape[1]*2) * .3
    q, qs, qg = reference.quantize(x.clone(), 16, weights_scaling_factor_2=a)
    expected_x = q.dequantize(dtype=x.dtype, scale=qs, double_scale=qg, block_sizes={-1: 16})
    assert torch.equal(activation_qdq(x, a), expected_x)
    checks['w4a4_activation_forward_exact_vs_modelopt'] = True
    probe = torch.randn(1, p.shape[0]) / p.shape[0]**.5
    for label, mode, dtype in [('w4a16', 'w4a16', torch.float32),
                               ('w4a4_ste', 'w4a4_ste', torch.float32),
                               ('w4a4_ste_bf16', 'w4a4_ste', torch.bfloat16)]:
        module = PackedNVFP4Linear(p, s, g, a, chunk_rows=256, activation=mode)
        value = x.to(dtype).detach().requires_grad_()
        started = time.perf_counter()
        output = module(value)
        grad, = torch.autograd.grad(output, value, probe.to(dtype))
        elapsed = time.perf_counter() - started
        expected_output = torch.empty_like(output)
        accumulation_dtype = torch.float32
        expected_grad = torch.zeros_like(value, dtype=accumulation_dtype)
        if mode == 'w4a4_ste':
            rq, rqs, rqg = reference.quantize(value.detach().clone(), 16, weights_scaling_factor_2=a)
            input_value = rq.dequantize(dtype=dtype, scale=rqs, double_scale=rqg, block_sizes={-1: 16})
            assert torch.equal(activation_qdq(value, a), input_value)
        else:
            input_value = value.detach()
        for first in range(0, p.shape[0], 256):
            last = min(first+256, p.shape[0])
            weight = reference_weight(first, last, value.dtype)
            expected_output[:, first:last] = input_value @ weight.T
            expected_grad += probe[:, first:last].to(dtype).float() @ weight.float()
        assert torch.equal(output, expected_output)
        assert torch.equal(grad, expected_grad.to(dtype))
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
        assert not list(module.parameters())
        checks[label] = dict(forward_exact_vs_reference=True, input_vjp_exact_vs_reference=True,
            gradient_norm=grad.norm().item(), output_norm=output.norm().item(),
            frozen_packed_buffers=True, dense_weight_retained=False,
            one_cpu_projection_forward_and_vjp_seconds=elapsed)
    small = PackedNVFP4Linear(p[:8], s[:8], g, a, chunk_rows=3, activation='w4a16')
    value = torch.randn(1, small.in_features, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(small, (value,), fast_mode=True)
    assert torch.autograd.gradgradcheck(small, (value,), fast_mode=True)
    checks['w4a16_real_rows_first_and_second_order_finite_difference'] = True
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    bank = torch.nn.Parameter(torch.randn(128, 4))
    host = torch.nn.Linear(16, p.shape[1]*2, bias=False).requires_grad_(False)
    saved_host = host.weight.detach().clone()
    branch = HNDLParticleProjection('real_nvfp4_input_probe', host, bank, 8, 8, attach=False)
    with torch.no_grad():
        branch.lora_up.weight.normal_(std=.02)
    branch.multiplier = 1.
    module = PackedNVFP4Linear(p, s, g, a, chunk_rows=256, activation='w4a4_ste')
    inputs = torch.randn(1, 1, 16)
    result = module(branch(inputs))
    (result * probe).sum().backward()
    assert bank.grad is not None and torch.isfinite(bank.grad).all() and torch.count_nonzero(bank.grad)
    assert torch.equal(saved_host, host.weight) and host.weight.grad is None
    checks['shared_hndl_particle_bank_through_real_nvfp4_projection'] = dict(
        finite_nonzero_gradient=True, gradient_norm=bank.grad.norm().item(),
        upstream_host='synthetic frozen 16->5120 projection, not full pretrained attention')
    with torch.no_grad():
        clean = module(host(inputs))
        branch.multiplier = 0.
        off = module(branch(inputs))
        branch.multiplier = -1.
        negative = module(branch(inputs))
    assert torch.equal(clean, off)
    assert not torch.equal(result, off) and not torch.equal(result, negative)
    checks['real_projection_with_upstream_shared_branch_strengths'] = dict(
        exact_clean_base_at_zero=True, nonzero_edit=True, signed_strength_difference=True)
    assert before == {n: digest(t) for n, t in tensors.items()}
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak_rss > 2 * 1024**3:
        raise RuntimeError('CPU probe exceeded proposed own 2 GiB peak budget')
    report = dict(schema='ripple-real-nvfp4-projection-probe-v1', model=origin['model'],
        revision=origin['revision'], logical_projection_shape=[p.shape[0], p.shape[1]*2],
        physical_packed_shape=list(p.shape), tensor_sha256=before,
        modelopt_reference_commit='43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a',
        modelopt_source_sha256=hashlib.sha256(args.modelopt_source.read_bytes()).hexdigest(),
        checkpoint_scales=dict(weight_scale_2=g.item(), input_scale=a.item()), checks=checks,
        cpu_peak_rss_bytes=peak_rss, host_available_at_admission_bytes=available,
        peer_global_reserve_bytes=47*1024**3, gpu_used=False,
        activation_gradient='explicit identity straight-through estimator, as ModelOpt pass_through_bwd=True; not derivative of rounding',
        qualification='one complete real gate projection only; no full-model or GPU qualification')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
