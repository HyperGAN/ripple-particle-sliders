"""Complete original token contexts and immutable first teacher observations."""
from pathlib import Path
import hashlib,json,os,uuid
import torch
from particle_sliders.pg09_audio import AudioRow,PreparedAudioResponse,audio_conditions
from .contracts import digest,file_hash

def read_rows(path,vocab):
    values=json.loads(Path(path).read_text())
    if not isinstance(values,list)or not values:raise ValueError('nonempty neutral/positive token rows required')
    rows=[]
    for value in values:
        if not isinstance(value,dict)or set(value)!={'neutral_ids','positive_ids'}:raise ValueError('exact matched neutral/positive ids required')
        for ids in value.values():
            if not isinstance(ids,list)or not 1<=len(ids)<=128 or any(type(v)is not int or not 0<=v<vocab for v in ids):raise ValueError('complete unpadded in-vocabulary context<=128; never truncate')
        if value['neutral_ids']==value['positive_ids']:raise ValueError('distinct matched teacher/student prompts required')
        rows.append(AudioRow(tuple(value['neutral_ids']),tuple(value['positive_ids'])))
    return rows

def record(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=file_hash(path))

def teachers(native,options,construction):
    path=Path(options['teacher_cache']);rows=read_rows(options['training_file'],native.model.config.vocab_size)
    identity=dict(schema='Ripple095-original-native-teachers-v1',manifest_sha256=options['cache_manifest_sha256'],
        rows=[dict(neutral_ids=list(r.neutral_ids),positive_ids=list(r.positive_ids))for r in rows],
        numerical_profile=native.runtime.identity['numerical_profile'])
    if path.exists():
        if construction.mode=='resume'and record(path)!=json.loads(construction.teacher_cache_json):raise ValueError('exact saved teacher cache bytes required; no regeneration')
        packet=torch.load(path,map_location='cpu',weights_only=True)
        if set(packet)!={'identity','observations'}or packet['identity']!=identity or len(packet['observations'])!=len(rows):raise ValueError('saved teacher cache native context/Source differs')
        prepared=[]
        for row,value in zip(rows,packet['observations']):
            if set(value)!={'target','neutral'}or any(not isinstance(v,torch.Tensor)or v.shape!=(1,5120)or v.dtype!=torch.float32 or v.requires_grad or not torch.isfinite(v).all()for v in value.values()):raise ValueError('actual full native5120 frozen teacher values required')
            prepared.append(PreparedAudioResponse(row,value['target'].to(native.runtime.device),value['neutral'].to(native.runtime.device)))
    else:
        if construction.mode!='fresh':raise ValueError('missing original teacher cache on resume; no repeated forwards')
        prepared=[native.prepare(row)for row in rows]
        value=dict(identity=identity,observations=[dict(target=p.target.detach().cpu(),neutral=p.neutral.detach().cpu())for p in prepared])
        temporary=path.with_name(path.name+'.complete-'+uuid.uuid4().hex)
        try:
            with temporary.open('xb')as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
            os.link(temporary,path);temporary.unlink()
            fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        finally:
            if temporary.exists():temporary.unlink()
    native.teacher_cache=record(path)
    return prepared,audio_conditions(prepared)
