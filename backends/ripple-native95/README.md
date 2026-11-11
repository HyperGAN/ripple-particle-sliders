# Ripple native95 owner API

This opt-in subproject exposes the captured `ripple_native95` backend for the full original Ripple-65B Preview NVFP4 revision. It leaves the repository's historical `ripple_particle_sliders` package, commands, defaults and cohorts unchanged. Use a separate environment: this profile requires Transformers 4.57.6, whereas the historical root package pins a different stack.

The backend retains all 80 blocks, all 320 Q/K/V/O adapter branches, rank/alpha 8, 128×4 particles, full 5120-wide responses, chunk256 and the original W4A4 activation identity-STE / BF16 reference GEMM path. This is not native FP4 TensorCore arithmetic. Frozen model/cache files and tokenizer assets are not distributed here. Source exposure and the retained small software observations do not qualify a future installed wheel, every full native workload, or output quality.

## Install the separate profile

From this repository checkout, in an isolated compatible environment:

```sh
python -m pip install ./backends/ripple-native95
```

The subproject pins ParticleGAN 0.9.5 and the exact public core `f7001a821c89df6c2e8473da4f6914db45516aeb` with its `fibo-hndl` extra. That core changes packaging metadata from parent `41f6066f2d2644902670d5eb736de052e1a70553`; the twelve required core Source leaves match `current_core.json` byte for byte. The unchanged lock deliberately records the parent runtime identity and core version 0.14.1. A newer version number alone is insufficient. Do not copy modules onto `PYTHONPATH`, edit guards, or mix this profile with the root package's dependencies.

## Supply authenticated offline inputs and a real owner

The caller supplies the complete original model tensor-cache directory and its exact `manifest.json` SHA256, authenticated tokenizer assets, the original matched token-row JSON, and an already completed matching teacher cache. Token rows are a JSON list of objects with exactly `neutral_ids` and `positive_ids`; each is a distinct, complete in-vocabulary sequence of 1–128 tokens. The teacher cache has schema `Ripple095-original-native-teachers-v1`, native Source/numerical identity and one full FP32 `(1,5120)` neutral/target pair per row. Do not reconstruct or relabel teacher results from an incomplete attempt.

The original ordinary factory can collect teachers on a *fresh* construction if its teacher file is absent. The examples below require the completed file before opening. Strict FULL restore refuses a missing or changed cache. No teacher collector is invoked by these instructions.

CUDA requires a runner-owned `OwnedCUDAAdmission`. The runner acquires and retains two real physical shared leases plus one role-exclusive lease, authenticates its own receipt, maps the logical device to the actual physical UUID, and supplies a live checker covering its caps, floors, lock modes and lifetime. The backend checks held FD/path identities and the actual CUDA UUID; it does not create resource admission. Keep the admission context and every lease alive through construction, work and close. Receipt field names describe evidence, not permission to invent it.

```python
from ripple_native95.ownership import HeldLease, OwnedCUDAAdmission

# Every value below comes from your runner's genuine admitted lifetime.
owner = OwnedCUDAAdmission(
    device=logical_device, physical_uuid=physical_uuid,
    owner_receipt=actual_owner_receipt,  # {path, bytes, sha256}
    physical_shared=(HeldLease(physical_lease_path_0, held_shared_fd_0),
                     HeldLease(physical_lease_path_1, held_shared_fd_1)),
    role_exclusive=HeldLease(role_lease_path, held_exclusive_fd),
    check_live=check_actual_owned_lifetime,
)
```

These are owner inputs supplied by your application, not literal grant files or a standalone launcher provided by this package. No automatic GPU choice, resource claim or foreign-job change occurs.

## Fresh training and exact FULL restore

Resolve fresh versus restore before opening. Keep the exact configuration, paths, original Source and cohort for a FULL restore. Fresh initialization uses the official scoped ParticleGAN initializer before optimizers, preserving original Parameter objects, declared host scales, zero-UP weights and the original unit-standard-deviation R2Normal bank. Restore validates the original receipt and skips fresh initialization. EMA is explicitly disabled for adapter, critic and bank weights; scalar optimizer statistics remain unchanged.

```python
from pathlib import Path
from ripple_native95 import factory
from ripple_native95.ownership import admitted

teacher_path = Path(completed_teacher_cache).absolute()
if not teacher_path.is_file():
    raise ValueError("an authentic completed teacher cache is required")

cfg = factory.config(
    cohort_id,
    cache_dir=str(Path(original_model_cache).absolute()),
    cache_manifest_sha256=authenticated_manifest_sha256,
    training_file=str(Path(matched_token_rows).absolute()),
    teacher_cache=str(teacher_path), device=owner.device,
    steps=128, batch=2, seed=7, ema_decay=None,
    lr=.012, d_lr_mult=1.5, prior_lr_mult=2.5,
)

# resume_FULL is None for a genuinely new cohort, otherwise its exact typed
# whole-game FULL packet/path. Never substitute a RAW artifact here.
with admitted(owner):
    with factory.open_training(cfg, resume=resume_FULL) as session:
        session.train(stop_after=stop_after)  # e.g. 32 within the fixed horizon128
        FULL = session.checkpoint(new_full_path)
        RAW = session.export_inference(new_raw_path, selection="raw")
```

`checkpoint()` returns the typed whole-game packet: game/critic/particles, optimizers and scalar statistics, RNG, recipe/config, cache and initialization identity. Resume it through the same factory/config and continue only genuinely missing updates. `export_inference()` returns the distinct native-only RAW artifact; it reconstructs no game, critic, teacher rows or initializer. Keep new destinations and durable call/commit accounting in the owning runner. The rates shown are a declared starting recipe, not a completed cross-rate recommendation; see [LEARNING-RATES.md](LEARNING-RATES.md).

## RAW finished-text inference

Use the additive public text facade. The captured core's generic `.render()` passes an `images=None` keyword that the unchanged Ripple text callbacks do not accept. The facade follows the existing successful text call sequence while retaining the original backend/factory identity, core observation/RNG restoration and retryable cleanup. It lives in the sibling `ripple_native95_public` namespace so it cannot change the original backend's recursive Source identity.

```python
from ripple_native95_public.text_api import open_text_inference

request = dict(ids=complete_prompt_token_ids, max_new_tokens=64)
with admitted(owner):
    with open_text_inference(
        RAW,
        options=dict(cache_dir=cfg.options["cache_dir"],
                     cache_manifest_sha256=cfg.options["cache_manifest_sha256"],
                     device=owner.device),
        strength=1., native_request=request,
    ) as inference:
        output = inference.render(strength=1., seed=7, **request)
        output.save(new_text_path)  # also writes the native JSON sidecar
```

The complete prompt plus generation budget must fit 128 tokens, without truncation. Native generation uses the complete original vocabulary head and greedy argmax, including literal EOS; strengths are 0–1. Non-`None` image references, including an empty list, are refused before allocation/work. The facade neither retags RAW nor blends an older adapter. Keep its context manager so failures invoke original session cleanup; a failed close remains retryable and blocks further work until cleanup succeeds.

## Saved campaign factory identities

The [saved cached-only profile assets](profiles/saved-full80-cached-only-v1/README.md) preserve the exact `cache_factory.py` `c7c5141a…` and `durable_teachers.py` `b3401177…` used by the captured execution composition. They are source assets, not replacement default backends. A FULL manifest made by that factory is not interchangeable with the ordinary factory above. RAW continues to use the original inference factory identity. No saved manifest, Source digest, receipt or checkpoint should be rewritten to make a different factory appear compatible.

## New facade software cases

All four new synthetic CPU cases in `tests/fresh95_text_owner_check.py` passed on 2026-11-11 against the genuine installed wheel, from a neutral directory with exact Source/RECORD checks and no source `PYTHONPATH`. See the [software validation record](validation/installed-owner-api-20261111.json). They cover forwarding/`None` filtering and original owner identity, pre-work reference refusal, actual core observation restoration/close after callback failure, and retry after a failed close. They do not load Ripple weights, collect teachers or perform optimizer updates. The opt-in filename stays outside default pytest discovery.

With the compatible dependencies and the genuine wheel installed, a future owner may explicitly invoke the fixture from a neutral directory, without a source `PYTHONPATH`:

```sh
python -m pytest -c /dev/null /absolute/checkout/backends/ripple-native95/tests/fresh95_text_owner_check.py
```

The offline wheel build and installation completed; the installed helper checks above passed. Full pretrained/native execution through this public wheel remains unverified. The saved campaign runs retain their original execution owners and dependency composition.
