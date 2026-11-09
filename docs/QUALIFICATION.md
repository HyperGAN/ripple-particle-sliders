# Qualification boundary

Verified sources are pinned public files from
`ricursiveAI/Ripple-65B-Preview-NVFP4@99eea4477a001a41b86914e41feb4b2ed86105e5`:
README, LICENSE, config, model/config implementation, tensor index and the
`inference/local/model.py` and `spark.py` runtime source. Source snapshots and
inspection receipts remain in the parent task's `validation/public-metadata`.

The standalone package vendors only unchanged architecture/config code, metadata
and Apache-2.0 license. The CPU fixture uses the actual architecture with reduced
dimensions and vocabulary, five layers (four local, one global), sliding window
four, gated attention, PeriNorm, value residuals, embedding scale and MLP clamps.
Its arithmetic is FP32 and its weights are random. Tests check architecture and
shared game behavior; they do not qualify pretrained NVFP4 inference or training.

The upstream HF implementation is differentiable for ordinary dense parameters.
The experimental packed projection now consumes one complete real ModelOpt
matrix and its frozen scales with verified CPU forward values and VJPs. That
does not establish a complete model loader. The official Spark
implementation is an SGLang inference engine with fused kernels and no-grad
forward. Transformers' `NVFP4Config` path is not interchangeable with this
prequantized ModelOpt format. The reference path does not remove no-grad from
SGLang or substitute BF16 source weights. It decodes the checkpoint's original
packed bytes on demand and explicitly selects activation-gradient semantics.

## Experimental packed reference (0.2.0)

`tools/fetch_nvfp4_probe.py` limits total HTTP ranges to 128 MiB, verifies exact
206 intervals, pins the checkpoint revision, and records original byte hashes.
It fetched all six small headers and the complete first gate projection/scales,
118,180,192 bytes total. `tools/probe_real_nvfp4.py` verifies byte hashes, compares
with the unchanged ModelOpt 0.46.0 quantize/dequantize methods, checks a full
projection's forward/VJP and real-row first/second-order finite differences,
then differentiates a shared HNDL particle bank through that real projection.
Its upstream projection is synthetic and is labeled as such. The complete
pretrained Ripple attention/MLP stack has not been run.

W4A4 rounding uses an identity STE as in ModelOpt's pass-through backward option.
Finite differences are appropriate for the linear W4A16 policy and are not
presented as proof of a nonzero true derivative of quantized activation rounding.
Only packed buffers are saved by the streamed linear autograd function. The
paired transposed operator recomputes decoding during higher-order backward.
No ParticleGAN objective or auxiliary loss is introduced by this experiment.

`tools/plan_nvfp4_resources.py` derives storage from all six actual shard headers
and adapter sizes from the real HNDL builder. Total checkpoint storage is
57,213,636,640 bytes; one packed/BF16 block is about 689.5 MB. A candidate single
A6000 streaming plan requests 4 GiB GPU and 3 GiB host RSS, with 128 tokens,
one active block, chunked decode, checkpoint recomputation and lazy embedding
rows. These are unmeasured full-model planning limits. A first representative
GPU projection test would request 2 GiB GPU, 3 GiB host RSS and 600 seconds from
the existing scheduler. No lease or job was started. Preserve the 5 GiB GPU free
floor and 47 GiB peer/global host reserve. The actual resource-plan JSON remains
in the parent task's `validation/nvfp4-investigation` directory.

Before enabling native training, verify all of the following under the existing
single scheduler and resource guardian:

1. A pinned loader consumes the exact six packed shards with static W4A4 scale
   semantics preserved; all MLPs remain NVFP4 and attention remains BF16.
2. Its kernels and model partitioning support the approved hardware. Bound host
   load peaks, KV state, activations, adapter/critic memory and optimizer state.
3. Nonzero gradients reach early and late attention adapters and the one shared
   particle bank through every quantized MLP; base parameters stay frozen.
4. Strength-zero output equals the same unmodified quantized base. Signed
   strength, repeated backward and raw/EMA export preserve the intended response.
5. A split/resume run matches uninterrupted updates, and checkpoint identity
   rejects changed model, runtime, scales, shard hashes, rows and device plan.
6. A real prompt/response evaluation establishes useful language behavior.

No GPU admission was requested because the shipped runtime cannot satisfy these
prerequisites on the observed SM86 hardware. Existing scheduler ownership and all
foreign jobs are preserved. The historical qualification did not publish code, accept gated terms or deploy.
The later source-only publication is documented separately in PUBLICATION.md.
