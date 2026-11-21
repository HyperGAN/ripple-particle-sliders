# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| NVFP4 file-backed hidden-response training | 1 | 10 | 12 | one CUDA GPU; batch 2; ≤ 128 tokens; packed MLP block16 FP8 scales, BF16 dense attention; all 80 blocks checkpointed; chunk256 |
| Native finished-text sampling | 1 | 6 | 12 | batch 1; entire prompt+new≤ 128; no KV; one block/head loaded at a time; full vocabulary |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.11.1 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **ricursiveAI/Ripple-65B-Preview-NVFP4**, revision **`99eea4477a001a41b86914e41feb4b2ed86105e5`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| NVFP4 file-backed hidden-response training | one_streamed_block: 0.6421; row_decode_temporaries: 0.1562; trainable_transient: 1.7739; checkpoint_activations: 0.9961; workspace: 6.0000 |
| Native finished-text sampling | max_block_or_full_head: 0.9570; adapter: 0.1109; prefill: 0.0586; workspace: 4.0000 |

Detailed arithmetic and assumptions:

- **MLP original elements**: `50331648000`.
- **packed MLP storage formula**: `Q/2 +Q/16 FP8 scales +small FP32 global/input scalars`.
- **one block bytes upper**: `689459248`.
- **decoded chunk temporary bytes**: `167772160`.
- **architecture checkpoint storage estimate bytes**: `57211944960`.
- **LoRA elements**: `27525120`.
- **projections**: `320`.
- **generator plus critic elements**: `29761889`.
- **host loading**: `two complete current-block buffers +two full vocabulary-head buffers +8 GiB +exports; complete checkpoint may additionally occupy reclaimable OS cache`.

Scope and limits:

- This native profile streams original packed per-block files. A65B dense model is not persistently resident. Only MLPs are NVFP4; attention, embeddings and head stay BF16.
- Execution is reference W4A4 straight-through arithmetic with per-row dequantization, not a claim of NVFP4 Tensor Core kernels. Packed format uses one E4M3 scale per16original weights, not the block64 FP4 formula.
- Finished text recomputes full prefixes and explicitly uses no KV cache. All128 tokens include the requested completion. Keep the Source-required shared checkpoint wrappers during gradient training so frozen block tensors can be released between recomputations.
- The full on-disk checkpoint/cache estimate is separate from resident memory. Non-streaming or eager full-model dequantization invalidates these budgets.

Source facts (read as text; no model execution):

- [backends/ripple-native95/src/ripple_native95/contracts.py](backends/ripple-native95/src/ripple_native95/contracts.py)
- [backends/ripple-native95/src/ripple_native95/streaming.py](backends/ripple-native95/src/ripple_native95/streaming.py)
- [backends/ripple-native95/src/ripple_native95/experimental_nvfp4.py](backends/ripple-native95/src/ripple_native95/experimental_nvfp4.py)
- [backends/ripple-native95/src/ripple_native95/finished_text.py](backends/ripple-native95/src/ripple_native95/finished_text.py)
- [backends/ripple-native95/src/ripple_native95/vendor/config.json](backends/ripple-native95/src/ripple_native95/vendor/config.json)
- [Pinned model metadata](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4/tree/99eea4477a001a41b86914e41feb4b2ed86105e5) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.11.1 release `d5dec2fe005305082acf382f0939a4bdb51c7131`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
