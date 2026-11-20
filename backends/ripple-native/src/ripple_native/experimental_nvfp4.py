"""Experimental packed NVFP4 CPU/reference execution, separate from native gate.

Weights remain the official packed bytes. Decode only the current output-row
chunk; forward and all VJPs recompute that chunk. No dense weight copy is kept.
The W4A4 forward uses the checkpoint's global activation scale and runtime
block-16 E4M3 scales, with explicit identity straight-through differentiation.
This is reference arithmetic, not a claim of native FP4 Tensor Core execution.
"""
import torch
from torch import nn

E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.)


def decode_rows(packed, block_scale, global_scale, *, dtype=torch.float32):
    """ModelOpt low-nibble-first E2M1 times E4M3 scale times FP32 scale."""
    if (packed.dtype != torch.uint8 or packed.ndim != 2
            or min(packed.shape) < 1
            or block_scale.dtype != torch.float8_e4m3fn
            or block_scale.shape != (packed.shape[0], packed.shape[1] // 8)
            or packed.shape[1] % 8 or global_scale.dtype != torch.float32
            or global_scale.numel() != 1):
        raise ValueError('expected packed E2M1, block-16 E4M3 and scalar FP32 scale')
    if packed.device != block_scale.device or packed.device != global_scale.device:
        raise ValueError('packed weights and both scales must share a device')
    table = torch.tensor(E2M1, device=packed.device, dtype=torch.float32)
    values = torch.empty((packed.shape[0], packed.shape[1] * 2),
                         device=packed.device, dtype=torch.float32)
    values[:, 0::2] = table[(packed & 15).long()]
    values[:, 1::2] = table[(packed >> 4).long()]
    scale = block_scale.float() * global_scale
    return (values.reshape(*block_scale.shape, 16) * scale.unsqueeze(-1)).flatten(-2).to(dtype)


@torch.no_grad()
def activation_qdq(x, global_scale):
    """Fixed global scale; dynamic group-16 FP8 scale and nearest-even E2M1.

    This function supplies the quantized forward only. Its caller explicitly
    chooses the identity STE; rounding has no mathematical nonzero derivative.
    """
    if x.shape[-1] % 16 or global_scale.numel() != 1:
        raise ValueError('block-16 activation shape and scalar scale required')
    if (not bool(torch.isfinite(global_scale).all()) or not bool((global_scale > 0).all())
            or not bool(torch.isfinite(x).all())):
        raise ValueError('finite activations and positive finite scale required')
    blocks = x.float().reshape(*x.shape[:-1], -1, 16)
    scale = blocks.abs().amax(-1) / (6. * global_scale)
    # Match ModelOpt's zero-block safety and E4M3 scaling conversion.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    scale = scale.clamp(min=2**-9, max=448.).to(torch.float8_e4m3fn).float() * global_scale
    normalized = blocks / scale.unsqueeze(-1)
    bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=x.device)
    magnitude = normalized.abs()
    ordinal = torch.bucketize(magnitude.contiguous(), bounds)
    for boundary in (.75, 1.75, 3.5):
        ordinal += (magnitude == boundary).long()
    code = ordinal + (normalized < 0).long() * 8
    values = torch.tensor(E2M1, device=x.device, dtype=torch.float32)[code]
    return (values * scale.unsqueeze(-1)).reshape_as(x).to(x.dtype)


class _PackedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scales, global_scale, chunk_rows, transpose):
        ctx.save_for_backward(packed, scales, global_scale)
        ctx.chunk_rows, ctx.transpose = chunk_rows, transpose
        n, k = packed.shape[0], packed.shape[1] * 2
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[-1] != (n if transpose else k):
            raise ValueError('projection input width differs')
        accumulation_dtype = (torch.float32 if transpose and x.dtype in
                              (torch.bfloat16, torch.float16) else x.dtype)
        out = torch.zeros(flat.shape[0], k if transpose else n,
                          device=x.device, dtype=accumulation_dtype)
        for first in range(0, n, chunk_rows):
            last = min(first + chunk_rows, n)
            weight = decode_rows(packed[first:last], scales[first:last], global_scale,
                                 dtype=x.dtype)
            if transpose:
                out.add_(flat[:, first:last].to(accumulation_dtype) @ weight.to(accumulation_dtype))
            else:
                out[:, first:last] = flat @ weight.T
        return out.to(x.dtype).reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        packed, scales, global_scale = ctx.saved_tensors
        # Use the paired streamed operator again instead of recording a graph
        # containing every decoded chunk. This also supports higher-order VJPs.
        grad_input = _PackedMatmul.apply(grad_output, packed, scales, global_scale,
                                        ctx.chunk_rows, not ctx.transpose)
        return grad_input, None, None, None, None, None


class PackedNVFP4Linear(nn.Module):
    """Frozen official packed weights; explicit W4A4-STE or W4A16 reference mode."""
    def __init__(self, packed, scales, global_scale, input_scale, *, chunk_rows=256,
                 activation='w4a4_ste'):
        super().__init__()
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError('positive integer chunk_rows required')
        if activation not in ('w4a4_ste', 'w4a16'):
            raise ValueError('explicit activation policy must be w4a4_ste or w4a16')
        if any(t.requires_grad for t in (packed, scales, global_scale, input_scale)):
            raise ValueError('packed checkpoint tensors and scales must remain frozen')
        # Validate actual bytes and shape through a bounded first-row decode.
        decode_rows(packed[:1], scales[:1], global_scale)
        if (packed.shape[0] != scales.shape[0] or input_scale.dtype != torch.float32
                or input_scale.numel() != 1 or input_scale.device != packed.device
                or not bool(torch.isfinite(global_scale).all()) or not bool((global_scale > 0).all())
                or not bool(torch.isfinite(scales.float()).all()) or not bool((scales.float() > 0).all())
                or not bool(torch.isfinite(input_scale).all()) or not bool((input_scale > 0).all())):
            raise ValueError('valid frozen shapes and positive finite scales required')
        for name, value in [('packed', packed), ('scales', scales),
                            ('global_scale', global_scale), ('input_scale', input_scale)]:
            self.register_buffer(name, value)
        self.chunk_rows, self.activation = chunk_rows, activation
        self.in_features, self.out_features = packed.shape[1] * 2, packed.shape[0]

    def forward(self, x):
        if (x.dtype not in (torch.float64, torch.float32, torch.bfloat16, torch.float16)
                or x.device != self.packed.device or x.shape[-1] != self.in_features):
            raise ValueError('floating activation with matching device and width required')
        if self.activation == 'w4a4_ste':
            quantized = activation_qdq(x, self.input_scale)
            x = quantized + (x - x.detach())
        return _PackedMatmul.apply(x, self.packed, self.scales, self.global_scale,
                                   self.chunk_rows, False)
