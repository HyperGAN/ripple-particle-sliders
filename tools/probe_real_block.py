"""One complete real pretrained decoder block: streamed/resident gradients on CPU."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
from contextlib import contextmanager

import torch
import psutil
from torch import nn
from torch.func import functional_call

from particle_sliders.pg09_backends import AdapterParameters, checkpoint_with_adapter_scale
from particle_sliders.fibo_hndl import HNDLParticleProjection
from ripple_particle_sliders.streaming import CURRENT, FileTensorStore, StreamedBlock
from ripple_particle_sliders.vendor.configuration_ricursive import RicursiveConfig


class Host(nn.Module):
    def __init__(self, config, block):
        super().__init__(); self.config, self.block = config, block


class BlockProbeBranches(nn.Module):
    """Numerical block VJP only; does not construct/admit a ParticleGAN response game.

    The released game's width>=170 gate is intentionally unchanged. These are
    the same shared HNDL projection primitives already qualified individually.
    """
    def __init__(self, block):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128,4))
        self.adapters = nn.ModuleList()
        self.hosts = []
        try:
            for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                module = getattr(block.native.self_attn, name)
                adapter = HNDLParticleProjection(name, module, self.particles, 8, 8, attach=False)
                self.adapters.append(adapter)
                self.hosts.append((module, module.forward, adapter))
                module.forward = adapter.forward
        except BaseException:
            self.close(); raise

    @contextmanager
    def scaled(self, strength):
        previous = [a.multiplier for a in self.adapters]
        try:
            for adapter in self.adapters: adapter.multiplier = float(strength)
            yield self
        finally:
            for adapter, value in zip(self.adapters, previous): adapter.multiplier = value

    def close(self):
        for module, original, adapter in self.hosts:
            adapter.multiplier = 0.
            if module.forward == original: continue
            if module.forward != adapter.forward: raise RuntimeError('probe forward ownership changed')
            module.forward = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(811)
    phases = []
    def milestone(name):
        row = dict(phase=name, rss_bytes=psutil.Process().memory_info().rss)
        phases.append(row); print(json.dumps(row), flush=True)
    milestone('imports')
    store = FileTensorStore(args.cache)
    if store.manifest['source'].get('revision') != '99eea4477a001a41b86914e41feb4b2ed86105e5':
        raise ValueError('wrong official checkpoint revision')
    config = RicursiveConfig.from_dict(store.manifest['config'])
    block = StreamedBlock(config, store.manifest['block_index'], store, chunk_rows=256)
    host = Host(config, block)
    # Two tokens, real BF16 attention and all three real packed MLP matrices.
    value = (torch.randn(1,2,config.hidden_size)*.2).to(torch.bfloat16).requires_grad_()
    seed = (torch.randn_like(value)*.2).requires_grad_()
    positions = torch.arange(2)
    options = dict(position_ids=positions.unsqueeze(0), cache_position=positions,
                   use_cache=False, value_residual=seed)
    started = time.perf_counter()
    with torch.no_grad(): clean = block(value, **options)[0].detach().clone()
    milestone('clean_forward_released')
    slider = BlockProbeBranches(block)
    with torch.no_grad():
        for adapter in slider.adapters: adapter.lora_up.weight.normal_(std=.02)
    network = AdapterParameters([(n,p) for n,p in slider.named_parameters() if p.requires_grad], slider.scaled)
    handle = checkpoint_with_adapter_scale(block, network)
    probe = torch.randn_like(value).float()/config.hidden_size**.5
    try:
        with torch.no_grad(), network.scaled(0.): off = block(value, **options)[0]
        assert torch.equal(clean, off)
        with network.scaled(1.): result = block(value, **options)[0]
        milestone('stream_forward_released')
        # Observe OFF after leaving strength context; backward must recover +1.
        with torch.no_grad(), network.scaled(0.): block(value, **options)
        (result.float()*probe).sum().backward()
        milestone('stream_backward_released')
        expected_output = result.detach().clone()
        expected_input, expected_seed = value.grad.clone(), seed.grad.clone()
        expected_gradients = {n:p.grad.clone() for n,p in slider.named_parameters()}
        assert torch.isfinite(slider.particles.grad).all() and torch.count_nonzero(slider.particles.grad)
        assert all(torch.isfinite(p.grad).all() for p in slider.parameters())
        assert all(p.device.type == 'meta' and not p.requires_grad for p in block.parameters())
        assert store.active_bytes == 0 and CURRENT.get() is None
        handle.close()
        # Same official architecture and bytes, ordinary resident autograd graph,
        # executed after streaming graph release so the two bases never coexist.
        value2 = value.detach().requires_grad_(); seed2 = seed.detach().requires_grad_()
        options2 = dict(options, value_residual=seed2)
        slider.zero_grad(set_to_none=True)
        with network.scaled(1.), store.block(block.index, 'cpu') as tensors:
            dense = {n:tensors[f'model.layers.{block.index}.{n}'] for n in block.dense_names}
            resident = functional_call(block.native, dense, (value2,), options2, strict=True)[0]
        (resident.float()*probe).sum().backward()
        milestone('resident_reference_backward')
        assert torch.equal(resident, expected_output)
        assert torch.equal(value2.grad, expected_input) and torch.equal(seed2.grad, expected_seed)
        for name, parameter in slider.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_gradients[name], atol=1e-6, rtol=1e-5)
        assert not torch.equal(clean, expected_output)
        report = dict(schema='ripple-real-block-stream-probe-v1', source=store.identity,
            block_index=block.index, qualification='one complete real local-attention block; not full model or GPU',
            all_real_block_tensors=len(store.entries), exact_block_bytes=store.manifest['total_block_bytes'],
            model_type='official PeriNorm/gated attention/value residuals/SWA plus 3 packed W4A4 MLP projections',
            native_dtype='BF16', context_tokens=2, dense_base_stays_meta=True,
            exact_streamed_vs_resident_forward=True, exact_input_and_value_seed_vjp=True,
            shared_adapter_gradients_match=True, particle_gradient_norm=slider.particles.grad.norm().item(),
            exact_clean_off_after_nonzero_adapters=True, nonzero_strength_edit=True,
            checkpoint_recompute_captures_strength=True,
            max_scoped_block_bytes=store.max_active_bytes,
            cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_seconds=time.perf_counter()-started, gpu_used=False,
            memory_phases=phases,
            particle_gan_response_game_executed=False,
            released_width_gate_preserved=True,
            activation_gradient='explicit identity STE; not an exact derivative of activation rounding')
        args.out.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))
    finally:
        handle.close(); slider.close(); store.close(); gc.collect()


if __name__=='__main__': main()
