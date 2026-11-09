# Experimental block streaming

`FileTensorStore`, `StreamedBlock`, `StreamingModel` and `StreamingBackend` provide
file-backed execution of the official Ripple architecture. This is experimental
numerical infrastructure. The public backend's full-model training gate remains
closed, and no language quality or GPU qualification is implied.

The architecture's ordinary parameters stay on `meta`. A block scope reads and
hashes that block's original tensors into owned memory. `torch.func.functional_call`
temporarily supplies its frozen dense attention/norm/value-residual parameters
and restores the meta parameters even if native execution fails. The MLP's three
projections use the verified packed E2M1/E4M3/global-scale implementation. Context
ownership prevents a projection from consuming another store's block state.

The original local/global routing, RoPE convention, Q/K norms, attention gate,
value-residual seed, PeriNorm, embedding scale and MLP clamps are preserved. The
hidden-readout host gathers selected embedding rows and omits the unused LM head.
It accepts complete unpadded sequences of at most 128 tokens, with no KV cache.
Embedding files are streamed through a hash once, then row reads check the same
file identity. Other tensors are hashed on every materialization. Manifest
descriptors are immutable snapshots and are included in model identity.

Gradients require the existing shared scale-capturing checkpoint wrapper around
each `StreamedBlock`. The shared AR adapter installs it through the backend's
`checkpoint_modules`. Recomputations re-open the original block bytes and use the
forward's captured slider strength. No copied ParticleGAN update loop or extra
training loss is added. W4A4 activation backward remains an explicit identity
straight-through estimator; it is not an exact derivative of activation rounding.

CPU fixtures compare the complete five-layer resident and streamed architectures,
including sliding windows longer than their local context, signed strengths,
shared-bank gradients after an intervening OFF observation, repeated backward,
cache corruption and native-exception cleanup. Real block probes use synthetic
hidden inputs and original pretrained weights, not real prompts or a full model.
The first five parity/cleanup tests passed before later snapshot hardening. The
expanded final suite is pending a fresh host window. A complete real block's
base forward returned in one guarded attempt, but no complete backward result is
qualified. Refused/stopped attempts and their partial memory measurements remain
separate from successful evidence.

## Preserved domain and resource gates

The pinned released ParticleGAN profile refuses flat continuous vectors of width
170 or greater. Ripple's native width is 5120. Neither response truncation,
field reshaping, a profile alias nor an engine patch is used to evade this gate.
Full native slider training needs an explicitly qualified continuous-vector
source profile and a distinct reviewed cohort, in addition to model/runtime and
resource qualification. Numerical HNDL projection/block VJP probes do not execute
or claim admission of a ParticleGAN response game.

The larger CPU probe is bounded by `tools/guard_stream_probe.py`. It preserves
47 GiB of peer/global host reserve, admits the complete declared own RSS budget
before spawning, tracks owned descendants and a 600-second deadline, and reaps
only its newly spawned process group if the future memory window stops fitting.
Failed/refused attempts retain separate immutable receipts. Lowering an own cap
does not lower the peer reserve, admit GPU work or alter foreign policy.

GPU execution requires the existing scheduler's valid route and admission. The
current route check did not establish one, so no GPU job or lease was created.
The proposed GPU-0 projection qualification remains 2 GiB VRAM, 3 GiB host RSS,
600 seconds; it must preserve foreign jobs and the 5 GiB GPU free floor.
