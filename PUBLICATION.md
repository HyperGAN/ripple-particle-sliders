# Public source scope and provenance

Source snapshot: `654b117a5b1a842a1e0f4ef723bb565b19821524`. Destination:
`HyperGAN/ripple-particle-sliders`. Original checkout/history is preserved.
This is a fresh publication snapshot, not a rewrite of the original branch.

## Qualification boundary

v0.3 streaming source is experimental; its full runtime suite was held for memory and is not qualified. Historical44 tests apply to v0.2; metadata audits and earlier packed-projection CPU work do not establish full-model NVFP4 or GPU qualification.

## Sanitation and retained behavior

Private adaptation paths are replaced by verified Gemma source identity. All four vendored upstream Apache files remain byte-identical to their recorded hashes. Hardware discussion is generalized; execution/admission guards remain unchanged.

Automated CI is not configured. The existing publication credential cannot add
GitHub Actions workflow files, and no credential or permission change is made.
Local focused CPU checks, when completed, are listed separately below; they are
not native execution, quality, new training admission or a hosted CI pass.
No model weights, deployment, GPU jobs, leases or other workloads are changed.

## Current focused checks — 2026-11-09

6 passed, 7 skipped, 20 subtests passed in the import-free metadata lane. Full v0.3 runtime/streamed-block/native tests were deliberately not run; historical44 remains a v0.2 result.
