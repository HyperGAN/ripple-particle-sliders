# Captured cached-only full80 factory profile

These two Python files are literal Source assets with their original bytes:

- `cache_factory.py`: SHA256 `c7c5141a46252714a7bf4aa01ef365c637c57a3ab4cdee6da8d52a325f781b96`.
- `durable_teachers.py`: SHA256 `b34011776c560783a9e4e8c166a5f49690b01a07b3aea072802ab3fd5aa8b74c`.

They are included in the source distribution, outside both installed runtime namespaces. They are not auto-registered, placed on a module search path, or substituted for the ordinary `ripple_native95.factory`. This placement also preserves the native model's recursive runtime Source identity.

The original cache factory imports a top-level `durable_teachers` module. Its cached-only companion imports an execution-owned `common` API and authenticates `common.ROOT / execution-contract-v1.json` plus its exact `reused_teacher_cache` FileRecord. That execution contract/lifetime authority is an external part of the original saved profile, not a generic public input manufactured here. Do not copy a private grant path, invent a compatible `common`, or edit that file to make it runnable. These assets alone do not provide a portable restore launcher for existing campaign FULL states.

A FULL state records the actual prepare/factory Source identity; preserve its original composition and complete config/cache/init/RNG/optimizer state when restoring it. The ordinary documented factory can create its own fresh cohort and restore its own exact FULL packets with an authentic pre-existing matching cache. The original native RAW inference factory is unchanged and is reached through the public text facade. No manifest or artifact identity is transplanted.
