"""Execution-owned cached-only companion; original cached-row math, no collector.

The original cache_factory body stays literal and receives this independently
pinned Source companion. A missing/changed cache always refuses, including
fresh construction; no fallback can call an original teacher response.
"""
from pathlib import Path
import json
import torch
from particle_sliders.pg09_audio import PreparedAudioResponse,audio_conditions
from ripple_native95.data import read_rows,record
import common as c


def valid_tensor(value):
    return(isinstance(value,torch.Tensor)and value.shape==(1,5120)and value.dtype==torch.float32
        and value.device.type=='cpu'and not value.requires_grad and torch.isfinite(value).all().item())


def teachers(native,options,construction,check):
    contract=c.pinjson(c.record(c.ROOT/'execution-contract-v1.json'));expected=contract['reused_teacher_cache']
    c.require(options['teacher_cache']==expected['path'],'Only the already completed exact original teacher cache')
    check();c.pinned(expected)
    path=Path(options['teacher_cache']);rows=read_rows(options['training_file'],native.model.config.vocab_size)
    identity=dict(schema='Ripple095-original-native-teachers-v1',manifest_sha256=options['cache_manifest_sha256'],
        rows=[dict(neutral_ids=list(r.neutral_ids),positive_ids=list(r.positive_ids))for r in rows],
        numerical_profile=native.runtime.identity['numerical_profile'])
    if construction.mode=='resume'and record(path)!=json.loads(construction.teacher_cache_json):
        raise ValueError('Exact saved teacher bytes on resume, never regeneration')
    # Unconditional existing-cache load; there is intentionally no missing-cache
    # or fresh-teacher branch and no callsite for native._response here.
    packet=torch.load(path,map_location='cpu',weights_only=True)
    if set(packet)!={'identity','observations'}or packet['identity']!=identity or len(packet['observations'])!=len(rows):
        raise ValueError('Complete original cached teacher identity differs')
    prepared=[]
    for row,value in zip(rows,packet['observations']):
        if set(value)!={'target','neutral'}or not all(valid_tensor(v)for v in value.values()):
            raise ValueError('Complete frozen native5120 cached teachers required')
        prepared.append(PreparedAudioResponse(row,value['target'].to(native.runtime.device),value['neutral'].to(native.runtime.device)))
    c.pinned(expected);check();native.teacher_cache=record(path)
    c.require(native.teacher_cache==expected,'The exact old cache must remain unchanged')
    return prepared,audio_conditions(prepared)
