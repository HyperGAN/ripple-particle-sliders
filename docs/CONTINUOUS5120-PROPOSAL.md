# Explicit continuous width-5120 proposal

**Source-qualified artifact identified; Ripple runtime remains unqualified.**
This proposal does not activate training, install a package, alter the released
gate, run a model or grant scheduler admission. Version 0.3.0 remains a candidate
until its full tests and native qualifications pass. The existing 0.2.0 projection
evidence and 0.3.0 refused/stopped attempts retain their separate identities.

## Intended shared API

Ripple's final hidden state is a native continuous `[1,5120]` vector, not a
categorical vocabulary. The released ParticleGAN heuristic treats a two-dimensional
matrix of width >=170 as categorical. The shared product gate refuses that
interpretation before a wide hidden response can enter the game.

The existing explicit source profile selects the continuous law without changing
geometry. It retains the continuous RMS gradient normalization, smooth Born–Infeld
cap, gradient floor and logit gauge. The shared game already passes
`input_domain="continuous"` to `recipe.make_critic_penalty` when configured with
the qualified pair. No product penalty, loss or optimizer implementation is needed.

The pair must be explicit and complete:

```python
pg_source_profile = "source-801ae1e-continuous-v1"
penalty_input_domain = "continuous"
```

The ordinary `release-0.9.4/auto` default remains unchanged. Selecting `continuous`
against release bytes, using a profile alias, truncating to width169 or reshaping
the vector as an image is refused. Source/domain pairs are not inferred from
metadata version alone.

References: [upstream domain API commit](https://github.com/255BITS/ParticleGAN/commit/801ae1ead8f88b88eca8f5398137e3e1bc01ad36)
and the immutable shared core's `pg094_continuous.json`, `pg09.py` and
`pg09_audio.py` at `5d68473a098801ee1ab3ebbb12463daef51c6346`.

## Exact artifact and proposal

The machine-readable proposal is
`src/ripple_particle_sliders/profiles/continuous5120.json`. It declares the native
response shape, condition width5124, distinct candidate cohort and disabled runtime.

| Component | Identity |
|---|---|
| Shared core | `5d68473a098801ee1ab3ebbb12463daef51c6346`, version0.12.0 |
| ParticleGAN source | `801ae1ead8f88b88eca8f5398137e3e1bc01ad36`, metadata version0.9.4 |
| Candidate PG wheel SHA256 | `263f8edead2f972b87d03a93780e95929c8c03d2989f3770313b88cb3699c9d0` |
| Response | `[1,5120]`, `continuous_vector` |
| Penalty declaration | `continuous` |
| Cohort | `ripple-nvfp4-pg094-source801-continuous5120-v1-candidate` |

An existing 296,898-byte candidate archive matches that exact wheel hash and all
51 Python source hashes in the shared core manifest. The two changed source files,
`bispf.py` and `recipes.py`, also match bytes fetched directly from the immutable
upstream commit. No source overlay or reconstructed wheel is used as evidence.
The installed PG package still matches the released source and differs from the
candidate in exactly those two files. Nothing was installed or changed there.

Compared with the release, the candidate includes an earlier tightening of Recipe
construction as well as the domain API: new runs use `get_recipe`, restoration uses
`Recipe.from_dict`, and direct construction requires all resolved fields. The
shared core already calls `get_recipe(preset, **options)`. Metadata-only checks
confirm identical resolved fields for both `bispf_optimized` and `bispf` at the
same batch/step/particle settings. This is configuration parity, not runtime or
gradient parity of a native model.

## Audit without importing Torch

In an existing dependency environment, run:

```bash
ripple-continuous-profile-audit --candidate-wheel /path/to/particlegan-0.9.4-py3-none-any.whl
```

The audit checks the entire installed core closure and immutable Git origin,
the core's expected continuous manifest, actual installed PG Python sources and
the optional archive's whole-file hash, metadata and Python closure. It never
changes `sys.path` to activate candidate sources or imports the training engine.
Even a matching archive/tree reports `runtime_admitted=false` and
`training_enabled=false`. A source mismatch/refusal exits with code2; code0
means only that the requested source audit passed, never compute admission.

Current result: `HOLD_INSTALLED_SOURCE_MISMATCH`, with matching released source
and two candidate-file differences. The correct candidate archive passes its
independent source audit. Thirteen standard-library-only tests pass under a
128 MiB address-space cap, with under 40 MiB peak RSS. The positive source fixture
is an unimported temporary snapshot of the exact archive. It is not an installed
runtime. These checks do not count as a rerun of the pending 0.3.0 Tensor suite.

## Runtime wiring still required

Keep this as a separate explicit cohort and use a fresh isolated
qualification environment; obtain scheduler admission before actual compute. Do not
replace the released or archived environments. The exact candidate wheel or
immutable source installation must satisfy the shared gate's complete source
inventory; version0.9.4 by itself is insufficient.

The product factory must then propagate the same pair through every boundary:

1. `capabilities`, option validation and `verify_runtime` must describe and check
   the selected pair. The current `verify_runtime` explicitly probes release
   defaults, so installing candidate PG alone must still fail.
2. `RippleHNDLSlider` must pass the pair to shared `require_audio_width` before
   graph construction. `RippleResponse` must pass it to the shared AR adapter.
   Their current default-release behavior must stay unchanged.
3. The HNDL critic's runtime verification and `ConditionalPG094` construction
   must use that same pair; no product formula or alternate optimizer is added.
4. Training configuration and inference specification must persist both fields,
   including reconstruction before adapters attach. Native identity must retain
   the shared domain gate's actual source digest and the immutable model cache.
5. Checkpoints must reject a different source profile, penalty declaration,
   geometry, model/data/recipe identity or old-cohort packet. The shared game
   already binds these fields and refuses automatic migration.

This proposal deliberately does not half-enable that wiring. Full-model NVFP4
training, raw/EMA native inference, source-profile runtime conformance and the
complete block-streaming backward proof remain closed/pending.

## Scheduled qualification scope

First use synthetic native-width5120 response vectors in the exact isolated
candidate environment, preserving geometry and the shared routed bank. Check
the public continuous penalty's RMS norm, floor/cap/gauge and finite first/second
derivatives; both optimizer roles and bank gradients; frozen host state; exact
split/resume; raw/EMA exports; and cross-profile refusal before mutation. Bind
the complete resolved recipe and source identities. This is a proposed future
compute test, not an executed result.

Then resume the complete guarded real-block proof and remaining 0.3.0 tests.
Only the existing scheduler may admit GPU work. Preserve its peer reserves,
foreign jobs and current holds. W4A4 uses an explicit identity straight-through
activation-gradient estimator, not an exact derivative of rounding. No full-model
language-quality qualification follows from a source or synthetic-vector test.
