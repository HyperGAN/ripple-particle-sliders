"""Compute exact checkpoint storage and clearly labeled streaming estimates."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from particle_sliders.fibo_hndl import HNDLParticleProjection, verified_builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headers', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    by_dtype, by_layer = Counter(), Counter()
    tensor_bytes, mlp_logical = {}, 0
    for path in args.headers.glob('*.header.json'):
        for name, desc in json.loads(path.read_text()).items():
            if name == '__metadata__':
                continue
            size = desc['data_offsets'][1] - desc['data_offsets'][0]
            tensor_bytes[name] = size
            by_dtype[desc['dtype']] += size
            if name.startswith('model.layers.'):
                by_layer[int(name.split('.')[2])] += size
            if desc['dtype'] == 'U8':
                mlp_logical += size * 2
    assert len(by_layer) == 80 and len(set(by_layer.values())) == 1
    bank = torch.nn.Parameter(torch.zeros(128, 4))
    shapes = dict(q=(5120, 10240), k=(5120, 1024), v=(5120, 1024), o=(10240, 5120))
    per_branch = {}
    for key, (n_in, n_out) in shapes.items():
        host = torch.nn.Linear(n_in, n_out, bias=False, device='meta')
        branch = HNDLParticleProjection(key, host, bank, 8, 8, attach=False)
        per_branch[key] = sum(p.numel() for p in branch.parameters())
    adapter_parameters = sum(per_branch.values()) * 80 + bank.numel()
    builder, _ = verified_builder()
    graph = '''joined = concat(response, condition, axis=1)
features = linear(joined, 16, name="hidden")
features = silu(features)
linear(features, 1, name="score")
'''
    critic = builder(graph, input_shape={'response': ['B', 5120], 'condition': ['B', 5124]},
                     output_shape=['B', 1])
    critic_parameters = sum(p.numel() for p in critic.parameters())
    # Two FP32 optimizer moments are a planning allowance, not an optimizer assertion.
    state_bytes = (adapter_parameters + critic_parameters) * 4 * 4
    tokens = 128
    report = dict(schema='ripple-nvfp4-resource-plan-v1',
        exact_checkpoint_bytes=sum(by_dtype.values()), exact_dtype_bytes=dict(by_dtype),
        exact_packed_mlp_logical_weights=mlp_logical,
        full_dense_bf16_base_bytes=mlp_logical*2 + sum(v for k,v in by_dtype.items() if k not in ('U8','F8_E4M3','F32')),
        exact_one_block_packed_and_bf16_bytes=next(iter(by_layer.values())),
        exact_all_blocks_forward_payload_bytes=sum(by_layer.values()),
        lm_head_needed_for_hidden_response=False,
        embedding_strategy='Read only selected token rows; do not materialize full vocabulary',
        exact_embedding_tensor_bytes=tensor_bytes['model.embed_tokens.weight'],
        exact_lm_head_bytes=tensor_bytes['lm_head.weight'],
        adapter_parameters=adapter_parameters, per_attention_branch_parameters=per_branch,
        critic_parameters=critic_parameters,
        fp32_parameters_gradients_two_optimizer_moments_allowance_bytes=state_bytes,
        optional_fp32_ema_extra_bytes=adapter_parameters*4,
        streaming_estimate=dict(
            qualified=False, target='one admitted SM86 A6000 with dequantized BF16 GEMMs',
            max_tokens=tokens, output_row_chunk=256,
            saved_block_input_bf16_allowance_bytes=80*tokens*5120*2,
            largest_dense_bf16_row_chunk_bytes=256*40960*2,
            decoder_fp32_and_index_workspace_allowance_bytes=256*1024**2,
            proposed_full_model_gpu_reservation_bytes=4*1024**3,
            proposed_full_model_host_rss_reservation_bytes=3*1024**3,
            required_local_base_disk_bytes=sum(by_dtype.values()),
            extra_checkpoint_disk_allowance_bytes=2*1024**3,
            minimum_peer_global_host_reserve_bytes=47*1024**3,
            minimum_gpu_free_floor_bytes=5*1024**3,
            constraints=['Only active block resident; release on own forward/backward completion',
                'Recompute packed row decode in VJPs and capture shared slider scale during recomputation',
                'Re-evict mapped checkpoint pages from own mapping; do not retain whole-model packed buffers',
                'A6000 GEMMs use BF16 after exact official-byte decoding, not native FP4 Tensor Cores',
                'W4A4 needs explicit STE; W4A16 changes activation semantics and must remain a separate mode',
                'Disk payload about 55 GB per blockwise model pass; recomputation/backward can multiply it',
                'No full-model streaming loader, native GPU test or scheduler admission established']),
        proposed_next_projection_gpu_qualification=dict(
            physical_gpu=0, requested_gpu_bytes=2*1024**3, requested_host_rss_bytes=3*1024**3,
            deadline_seconds=600, model_projection_only=True, admission_established=False,
            foreign_job_control='none', gpu1_use=False))
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
