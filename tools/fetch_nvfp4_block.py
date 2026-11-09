"""Fetch a single complete official decoder block, capped at 1 GiB, using ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from fetch_nvfp4_probe import MODEL, REVISION, read_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headers', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-projection', type=Path)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.index < 80:
        raise ValueError('block index must be in [0,80)')
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.headers / 'receipt.json').read_text())
    if (receipt['model'], receipt['revision']) != (MODEL, REVISION):
        raise ValueError('wrong source header identity')
    all_entries = {}
    for shard, metadata in receipt['headers'].items():
        header = json.loads((args.headers / (shard + '.header.json')).read_text())
        all_entries.update({name: (shard, metadata['header_bytes'], desc)
                            for name, desc in header.items()
                            if name.startswith(f'model.layers.{args.index}.')})
    total = sum(desc['data_offsets'][1] - desc['data_offsets'][0]
                for _, _, desc in all_entries.values())
    if not all_entries or total > 1024**3:
        raise ValueError('single-block byte budget exceeded')
    manifest = dict(schema='ripple-raw-tensor-cache-v1',
        source=dict(model=MODEL, revision=REVISION, qualification='one complete official block only'),
        config=json.loads(args.config.read_text()), tensors={}, actual_network_bytes=0,
        whole_shards_downloaded=False, block_index=args.index, total_block_bytes=total)
    reuse = json.loads((args.reuse_projection / 'receipt.json').read_text()) if args.reuse_projection else {}
    for name, (shard, header_size, desc) in sorted(all_entries.items()):
        start, end = desc['data_offsets']
        length = end-start
        target = args.out / (name + '.bin')
        previous = reuse.get('tensors', {}).get(name)
        if previous is not None:
            source = args.reuse_projection / previous['file']
            if source.stat().st_size != length or hashlib.sha256(source.read_bytes()).hexdigest() != previous['sha256']:
                raise ValueError('reused projection differs')
            shutil.copyfile(source, target)
            digest = previous['sha256']
        else:
            data, _ = read_range(shard, 8+header_size+start, length)
            target.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            manifest['actual_network_bytes'] += length
        manifest['tensors'][name] = dict(file=target.name, dtype=desc['dtype'],
            shape=desc['shape'], bytes=length, sha256=digest)
        (args.out / 'manifest.partial.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(block_index=args.index, bytes=total,
        network_bytes=manifest['actual_network_bytes'], tensors=len(all_entries)), indent=2))


if __name__ == '__main__':
    main()
