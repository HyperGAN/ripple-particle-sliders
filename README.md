# Ripple particle sliders

**Experimental source publication.** v0.3 streaming source is experimental; its full runtime suite was held for memory and is not qualified. Historical44 tests apply to v0.2; metadata audits and earlier packed-projection CPU work do not establish full-model NVFP4 or GPU qualification.

See [PUBLICATION.md](PUBLICATION.md) for sanitation, lineage and current CPU/CI limits.


Experimental public source for `HyperGAN/ripple-particle-sliders`, targeting
[Ricursive's official Ripple-65B Preview NVFP4](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4)
at revision `99eea4477a001a41b86914e41feb4b2ed86105e5`.

**Full-model NVFP4 training is blocked.** The package currently qualifies the shared
ParticleGAN lifecycle on small random FP32 CPU instances of the unmodified official
Ripple architecture. These fixtures do not simulate NVFP4 or demonstrate model
quality. No pretrained weights are included or published. A separate CPU probe
downloaded one complete official packed projection and verified its decoding and
gradient path; it did not train the full model.

The NVFP4 artifact is 57,213,636,640 bytes across six shards. Its 80-layer dense
causal language model uses 64 sliding-window and 16 global attention blocks,
gated attention, value residuals and PeriNorm. ModelOpt static W4A4 group-16
quantization covers the MLP gate/up/down projections. Attention, embeddings and
LM head remain BF16. The proposed slider targets only attention Q/K/V/O;
attention gates and quantized MLPs remain unchanged.

The official Spark path requires DGX Spark/GB10 SM12.1 and decorates model forward
with `torch.no_grad()`. The historical hardware assessment used SM8.6 cards with 48 GiB per card. One
such card cannot hold the packed checkpoint; summing cards does not establish a working
multi-device loader or differentiable quantized kernels. No GPU job was launched
and no scheduler reservation or unrelated process was changed.

The package rejects full-model training before loading weights or attaching
adapters. It never silently loads the BF16 checkpoint. Experimental on-demand
decoding uses the exact official NVFP4 bytes with higher-precision GEMMs. See
[the qualification notes](docs/QUALIFICATION.md).

## Real NVFP4 projection result

The experimental `ripple_particle_sliders.experimental_nvfp4.PackedNVFP4Linear`
keeps frozen packed E2M1 weights, E4M3 block-16 scales and FP32 global scales.
It decodes only an output-row chunk and recomputes chunks during backward,
including higher-order VJPs, without retaining a dense weight copy.
BF16/FP16 input-gradient contributions accumulate across chunks in FP32.

One complete real `model.layers.0.mlp.gate_proj` matrix, logical shape
`[40960, 5120]`, passed CPU checks against NVIDIA ModelOpt 0.46.0 at commit
`43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`. Its packed weight SHA256 is
`76005cad2cd801a2352a8254ecc94e3a15cd61cb5b4c16f7e160e50d4f05fb78`.
Both decoding and W4A4 activation quantization match that reference. Forward
values and input VJPs match; a shared HNDL particle bank receives finite nonzero
gradients through the real projection. The process peaked at 842,960,896 bytes
RSS. About 118.18 MB of checkpoint bytes/headers were fetched with bounded
HTTP ranges; no whole shard or full model was downloaded.

The default experimental `w4a4_ste` policy keeps quantized forward activations
and uses an explicit identity straight-through estimator for rounding, matching
ModelOpt's `pass_through_bwd=True` convention. That estimator is not the
mathematical derivative of rounding. The separate `w4a16` mode changes activation
semantics and is labeled accordingly. Neither is native FP4 Tensor Core execution.
The reference work used PyTorch 2.9.1+cu128 on CPU; CUDA execution remains untested.

This removes the supplied inference runtime's no-grad restriction from the
representative gradient path. A complete streamed model loader, A6000 execution,
full-model checkpoint identity and real language evaluation remain outstanding.
The public backend gate stays closed while those are unqualified.

## Block streaming (0.3.0)

This is a source candidate with final validation on hold for host memory. The
initial five architecture/parity tests passed; subsequent hardening and repeated
backward checks are prepared but the final suite has not yet run. Version 0.2.0
remains the last fully tested package (44 tests).

The experimental file-backed host now routes complete official decoder blocks
through temporary frozen parameters, the packed MLP path and the shared
scale-capturing checkpoint helper. Five-layer CPU fixtures compare resident and
streamed forwards and gradients, including local/global attention, value residuals
and context-window masking. See [streaming details and gates](docs/STREAMING.md).

One complete real block is cached separately for a guarded numerical probe.
The cache contains original bytes and is excluded from source distributions.
Full-model and GPU qualification remain pending. The pinned released
ParticleGAN profile also refuses native flat vectors of width >=170; Ripple's
5120-dimensional response requires an explicitly qualified continuous profile.
That restriction is preserved, with no response projection or profile substitution.
The real block's complete backward qualification has not finished: the host guard
refused or stopped attempts as the preserved memory window disappeared.

An [explicit width-5120 source proposal](docs/CONTINUOUS5120-PROPOSAL.md) now binds
the shared continuous-domain profile and an exact existing PG source archive.
Its import-free-of-Torch audit rejects the installed release as a candidate,
preserves the native vector shape and keeps training disabled. Thirteen separate
metadata/configuration tests pass; they do not qualify the pending 0.3.0 runtime.

## Install and inspect

Use Python 3.11–3.14 and a compatible PyTorch installation. Install this local
source and its immutable shared core dependency:

```bash
pip install '.[test]'
ripple-nvfp4-check
pytest -q
```

The shared command discovers the bounded profile:

```bash
particle-slider train --config examples/cpu-fixture.json \
  --profile ripple-cpu-fixture --out runs/cpu-check --check --no-reporting
```

`ripple-nvfp4-check` inspects bundled pinned metadata offline. It does not allocate
a model, download weights, infer hardware support, or grant scheduler admission.

## CPU architecture qualification

```python
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference
from ripple_particle_sliders.binding import config

rows = [
    {"neutral_ids": [1, 2], "positive_ids": [1, 3]},
    {"neutral_ids": [4, 5], "positive_ids": [4, 6]},
]
with open_response(config(fixture=True, fixture_width=16, steps=4), rows=rows) as run:
    run.train(stop_after=2)
    run.checkpoint("resume.pt")
    run.train()
    artifact = run.export_inference(selection="raw")
    expected = run.evaluate(strength=1)
with open_inference(artifact, operation="response") as probe:
    result = probe.response(ids=[1, 2], strength=1)
```

Token IDs here are synthetic fixture inputs, not tokenized coding instructions.
Strength zero follows the frozen architecture exactly; signed strengths use the
same shared routed particle bank. Raw/EMA inference export and full resumable
checkpoints belong to the shared core. This response backend returns a final-token
hidden vector, not generated text or an image.

## Formulation and ownership

Dependency direction is `ripple-particle-sliders -> particle-sliders-core ->
HyperGAN/HNDL and ParticleGAN`. The shared core owns the generator/critic game,
particle prior, row selection, optimizer, penalty, training checkpoints and inference
exports. This product supplies the frozen native host, attention attachment and
last-token response. No cross-entropy objective, reconstruction loss, preservation
loss, distillation or extra alignment objective is added.

The exact core Git origin and recursive source hashes are verified before hooks
attach. Checkpoints bind product source, upstream source and config, random base
weights, response protocol, rows, optimizers and RNG. CPU fixtures retain the
same shared `SessionConfig`, `ResponseBackend` and `InferenceBackend` interface
used by the Gemma/TinyQwen integrations. Independent source is MIT; the unmodified
vendored Ripple model code/config is Apache-2.0, with its license included.
See [PROVENANCE.json](PROVENANCE.json) and [publication scope](PUBLICATION.md).
This is public source, not a full-model training or native qualification release.
