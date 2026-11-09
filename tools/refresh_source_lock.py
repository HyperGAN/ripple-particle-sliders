"""Maintainer-only lock regeneration after reviewed product edits.

This updates product hashes only; it cannot authorize a new upstream/core origin.
"""
import json
from ripple_particle_sliders import binding

path = binding.ROOT / 'sources.lock.json'
lock = json.loads(path.read_text())
lock['product_sources'] = binding._package_sources(binding.ROOT, exclude=('sources.lock.json',))
lock['product_sources_index_sha256'] = binding._index_digest(lock['product_sources'])
path.write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
