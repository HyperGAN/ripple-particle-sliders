"""Fetch bounded public checkpoint byte ranges; never download a whole shard.

Default: headers only. --projection obtains one real gate projection and scales
(about 118 MB); no credentials, model execution, or CUDA use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request

MODEL = 'ricursiveAI/Ripple-65B-Preview-NVFP4'
REVISION = '99eea4477a001a41b86914e41feb4b2ed86105e5'
MAX_HEADER = 2 * 1024**2
MAX_TOTAL = 128 * 1024**2


def read_range(shard, start, length):
    if not 0 < length <= MAX_TOTAL or start < 0:
        raise ValueError('invalid or excessive bounded range')
    url = (f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{shard}'
           f'?download=true&probe_start={start}&probe_length={length}')
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start+length-1}',
                                                 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=45) as response:
        interval = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if (response.status != 206 or interval is None
                or tuple(map(int, interval.groups()[:2])) != (start, start + length - 1)):
            raise ValueError('server did not honor exact range; refusing whole-shard response')
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) != length:
            raise ValueError('unexpected range response length')
        data = response.read(length + 1)
        if len(data) != length:
            raise ValueError('short or oversized range response')
        return data, int(interval[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--projection', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(model=MODEL, revision=REVISION, headers={}, tensors={},
                   max_download_bytes=MAX_TOTAL, actual_download_bytes=0,
                   cuda_used=False, whole_shards_downloaded=False)
    for i in range(1, 7):
        shard = f'model-{i:05d}-of-00006.safetensors'
        lead, total = read_range(shard, 0, 8)
        size = struct.unpack('<Q', lead)[0]
        if not 0 < size <= MAX_HEADER:
            raise ValueError('invalid or excessive safetensors header')
        data, checked_total = read_range(shard, 8, size)
        if checked_total != total:
            raise ValueError('shard total size changed')
        header = json.loads(data)
        (args.out / (shard + '.header.json')).write_text(json.dumps(header, indent=2) + '\n')
        receipt['actual_download_bytes'] += 8 + size
        receipt['headers'][shard] = dict(header_bytes=size, file_bytes=total,
            header_sha256=hashlib.sha256(data).hexdigest())
        if args.projection and i == 1:
            prefix = 'model.layers.0.mlp.gate_proj'
            for suffix in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale'):
                name = f'{prefix}.{suffix}'
                desc = header[name]
                start, end = desc['data_offsets']
                length = end - start
                if receipt['actual_download_bytes'] + length > MAX_TOTAL:
                    raise ValueError('probe total byte budget exceeded')
                value, checked_total = read_range(shard, 8 + size + start, length)
                if checked_total != total:
                    raise ValueError('shard total size changed')
                (args.out / (name + '.bin')).write_bytes(value)
                receipt['actual_download_bytes'] += length
                receipt['tensors'][name] = dict(desc, file=name + '.bin', bytes=length,
                    sha256=hashlib.sha256(value).hexdigest())
        (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
