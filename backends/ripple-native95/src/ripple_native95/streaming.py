"""Experimental immutable file-backed Ripple blocks; no native training admission.

The official architecture stays on meta. Only the active block's frozen tensors
are materialized, and torch.func restores the meta parameters transactionally.
The shared core checkpoint wrapper must surround each block during autograd.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
from types import MappingProxyType

import torch
from torch import nn
from torch.func import functional_call

from .experimental_nvfp4 import PackedNVFP4Linear
from .vendor.modeling_ricursive import RicursiveDecoderLayer, RicursiveRMSNorm

CURRENT = ContextVar('ripple_stream_block', default=None)
DTYPES = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn,
          'BF16': torch.bfloat16, 'F32': torch.float32, 'F64': torch.float64}


def _stat_key(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class FileTensorStore:
    """Hash-checked raw checkpoint cache; memory scopes own only their tensors."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_bytes = (self.root / 'manifest.json').read_bytes()
        self._manifest = json.loads(self.manifest_bytes)
        if self._manifest.get('schema') != 'ripple-raw-tensor-cache-v1':
            raise ValueError('unsupported raw checkpoint cache')
        self.entries = MappingProxyType({name: MappingProxyType(dict(desc, shape=tuple(desc['shape'])))
                                        for name, desc in self._manifest['tensors'].items()})
        self.active_bytes = self.max_active_bytes = self.load_count = 0
        self._guard = threading.Lock()
        self._closed = False
        self._verified_files = {}

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    @property
    def identity(self):
        return dict(manifest_sha256=hashlib.sha256(self.manifest_bytes).hexdigest(),
                    source=deepcopy(self._manifest['source']), storage='original checkpoint bytes')

    def _descriptor(self, name):
        if self._closed:
            raise RuntimeError('tensor store closed')
        desc = self.entries[name]
        path = (self.root / desc['file']).resolve()
        if not path.is_relative_to(self.root) or desc['dtype'] not in DTYPES:
            raise ValueError('invalid cache path or dtype')
        shape = desc['shape']
        if not isinstance(shape, tuple) or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError('invalid cache tensor shape')
        size = torch.empty((), dtype=DTYPES[desc['dtype']]).element_size()
        for dim in shape:
            size *= dim
        if size != desc['bytes'] or not 0 < size <= 2 * 1024**3:
            raise ValueError('cache tensor byte size differs or exceeds 2 GiB')
        return desc, path

    def read(self, name, device):
        desc, path = self._descriptor(name)
        data = bytearray(desc['bytes'])
        with path.open('rb', buffering=0) as file:
            before = os.fstat(file.fileno())
            count = file.readinto(data)
            extra = file.read(1)
            after = os.fstat(file.fileno())
        if count != len(data) or extra or _stat_key(before) != _stat_key(after):
            raise ValueError('cache tensor changed or truncated during read')
        if hashlib.sha256(data).hexdigest() != desc['sha256']:
            raise ValueError('cache tensor hash differs')
        result = torch.frombuffer(data, dtype=DTYPES[desc['dtype']]).reshape(desc['shape'])
        self.load_count += 1
        return result.to(device=device)

    def rows(self, name, ids, device):
        """Hash a complete file once, then gather only requested embedding rows."""
        desc, path = self._descriptor(name)
        if len(desc['shape']) != 2 or ids.dtype != torch.long or ids.device.type != 'cpu':
            raise ValueError('row reads require a matrix and CPU int64 indices')
        count, width = desc['shape']
        if ids.numel() == 0 or bool((ids < 0).any()) or bool((ids >= count).any()):
            raise ValueError('invalid embedding token IDs')
        row_bytes = desc['bytes'] // count
        output = bytearray(ids.numel() * row_bytes)
        with path.open('rb', buffering=0) as file:
            before = os.fstat(file.fileno())
            signature = (before.st_dev, before.st_ino, before.st_size,
                         before.st_mtime_ns, before.st_ctime_ns)
            if before.st_size != desc['bytes']:
                raise ValueError('embedding file size differs')
            if self._verified_files.get(name) != signature:
                digest = hashlib.sha256()
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != desc['sha256']:
                    raise ValueError('embedding cache hash differs')
                self._verified_files[name] = signature
            for i, token in enumerate(ids.reshape(-1).tolist()):
                file.seek(token * row_bytes)
                target = memoryview(output)[i*row_bytes:(i+1)*row_bytes]
                if file.readinto(target) != row_bytes:
                    raise ValueError('short embedding row')
            if _stat_key(before) != _stat_key(os.fstat(file.fileno())):
                raise ValueError('embedding file changed during row reads')
        self.load_count += 1
        return torch.frombuffer(output, dtype=DTYPES[desc['dtype']]).reshape(*ids.shape, width).to(device)

    @contextmanager
    def block(self, index, device):
        if not self._guard.acquire(blocking=False):
            raise RuntimeError('tensor store already has an active block')
        tensors, token = {}, None
        try:
            prefix = f'model.layers.{index}.'
            names = sorted(name for name in self.entries if name.startswith(prefix))
            if not names:
                raise ValueError('requested block absent from cache')
            for name in names:
                tensors[name] = self.read(name, device)
                self.active_bytes += self.entries[name]['bytes']
                self.max_active_bytes = max(self.max_active_bytes, self.active_bytes)
            packet = dict(store=self, tensors=tensors)
            token = CURRENT.set(packet)
            yield tensors
        finally:
            if token is not None:
                CURRENT.reset(token)
            tensors.clear()
            self.active_bytes = 0
            self._guard.release()

    def close(self):
        if self._guard.locked():
            raise RuntimeError('cannot close an active block store')
        self._closed = True
        self._verified_files.clear()


class TransientPackedLinear(nn.Module):
    def __init__(self, store, name, n_in, n_out, chunk_rows, activation):
        super().__init__()
        self.store, self.name = store, name
        self.in_features, self.out_features = n_in, n_out
        self.chunk_rows, self.activation = chunk_rows, activation

    def forward(self, value):
        packet = CURRENT.get()
        if packet is None or packet['store'] is not self.store:
            raise RuntimeError('packed projection needs its owned active block scope')
        values = packet['tensors']
        projection = PackedNVFP4Linear(
            *(values[f'{self.name}.{key}'] for key in
              ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')),
            chunk_rows=self.chunk_rows, activation=self.activation)
        if (projection.in_features, projection.out_features) != (self.in_features, self.out_features):
            raise ValueError('packed projection architecture differs')
        return projection(value)


class StreamedBlock(nn.Module):
    def __init__(self, config, index, store, *, chunk_rows=256, activation='w4a4_ste'):
        super().__init__()
        self.index, self.store = index, store
        with torch.device('meta'):
            self.native = RicursiveDecoderLayer(config, index).eval().requires_grad_(False)
        for leaf in ('gate', 'up', 'down'):
            name = leaf + '_proj'
            original = getattr(self.native.mlp, name)
            setattr(self.native.mlp, name, TransientPackedLinear(store,
                f'model.layers.{index}.mlp.{name}', original.in_features,
                original.out_features, chunk_rows, activation))
        self.dense_names = tuple(name for name, _ in self.native.named_parameters())
        required = {f'model.layers.{index}.{name}' for name in self.dense_names}
        required |= {f'model.layers.{index}.mlp.{leaf}_proj.{suffix}'
                     for leaf in ('gate', 'up', 'down')
                     for suffix in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')}
        if not required <= set(store.entries):
            raise ValueError('incomplete block cache')

    def forward(self, hidden_states, **kwargs):
        if torch.is_grad_enabled() and not getattr(self, '_pg09_scale_checkpoint', False):
            raise RuntimeError('streaming gradients require shared scale-capturing block checkpoint')
        with self.store.block(self.index, hidden_states.device) as values:
            dense = {name: values[f'model.layers.{self.index}.{name}'] for name in self.dense_names}
            if any(t.requires_grad for t in dense.values()):
                raise ValueError('streamed base tensors must remain frozen')
            return functional_call(self.native, dense, (hidden_states,), kwargs, strict=True)


class StreamingModel(nn.Module):
    """Text-only no-padding/no-KV-cache hidden readout with original layer routing."""
    def __init__(self, config, store, *, device='cpu', chunk_rows=256, activation='w4a4_ste'):
        super().__init__()
        self.config, self.store, self.device = config, store, torch.device(device)
        self.layers = nn.ModuleList(StreamedBlock(config, i, store,
            chunk_rows=chunk_rows, activation=activation) for i in range(config.num_hidden_layers))
        with torch.device('meta'):
            self.norm = RicursiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps).requires_grad_(False)
        if not {'model.embed_tokens.weight', 'model.norm.weight'} <= set(store.entries):
            raise ValueError('embedding/final norm missing from full hidden-readout cache')

    def hidden(self, ids):
        if (not isinstance(ids, (tuple, list)) or not 1 <= len(ids) <= 128
                or any(type(v) is not int or not 0 <= v < self.config.vocab_size for v in ids)):
            raise ValueError('complete unpadded token IDs with context <=128 required')
        token_ids = torch.tensor([ids], dtype=torch.long, device='cpu')
        value = self.store.rows('model.embed_tokens.weight', token_ids, self.device)
        if self.config.embed_scale is not None:
            value = value * self.config.embed_scale
        seed = value if self.config.use_valres else None
        positions = torch.arange(len(ids), device=self.device)
        for layer in self.layers:
            value = layer(value, position_ids=positions.unsqueeze(0), cache_position=positions,
                          use_cache=False, value_residual=seed)[0]
        weight = self.store.read('model.norm.weight', self.device)
        return functional_call(self.norm, {'weight': weight}, (value,), strict=True)


class StreamingBackend:
    """Experimental host for the same shared AR response/scale-checkpoint adapter."""
    def __init__(self, store, *, device='cpu', chunk_rows=256, activation='w4a4_ste'):
        from .vendor.configuration_ricursive import RicursiveConfig
        self.model = StreamingModel(RicursiveConfig.from_dict(store.manifest['config']), store,
                                    device=device, chunk_rows=chunk_rows, activation=activation)
        self.checkpoint_modules = tuple(self.model.layers)

    def hidden(self, ids):
        return self.model.hidden(ids)
