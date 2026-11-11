"""Full original vocabulary-head greedy AR text; no hidden-vector decoding proxy."""
from dataclasses import dataclass
from pathlib import Path
import json,math,os
import torch
import torch.nn.functional as F
from .ownership import require_owner

def validate_request(*,ids,max_new_tokens=64,strength=1.,seed=0,output_type='text'):
    if output_type!='text':raise ValueError('native text output required')
    if not isinstance(ids,(tuple,list))or not ids or any(type(v)is not int or v<0 for v in ids):raise ValueError('complete native prompt ids required')
    if type(max_new_tokens)is not int or not 1<=max_new_tokens<128 or len(ids)+max_new_tokens>128:raise ValueError('entire prompt+generation<=128; no truncation')
    if type(seed)is not int or not 0<=seed<2**63-100:raise ValueError('literal seed required')
    if type(strength)not in(int,float)or not math.isfinite(strength)or not 0<=strength<=1:raise ValueError('finite strength0..1 required')

@dataclass(frozen=True)
class TextOutput:
    text:str
    native:dict
    def save(self,path):
        path=Path(path);sidecar=Path(str(path)+'.native.json')
        if path.exists()or sidecar.exists():raise ValueError('fresh text/receipt destinations required')
        with path.open('x')as stream:stream.write(self.text);stream.flush();os.fsync(stream.fileno())
        with sidecar.open('x')as stream:json.dump(self.native,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

@torch.no_grad()
def generate(native,*,ids,max_new_tokens=64,strength=1.,seed=0,output_type='text'):
    validate_request(ids=ids,max_new_tokens=max_new_tokens,strength=strength,seed=seed,output_type=output_type)
    runtime=native.runtime;require_owner(str(runtime.device));tokenizer=runtime.load_tokenizer();config=runtime.model.config
    if any(v>=config.vocab_size for v in ids):raise ValueError('original native vocabulary token ids required')
    eos=config.eos_token_id;eos=[eos]if type(eos)is int else eos
    if not isinstance(eos,(list,tuple))or not eos or any(type(v)is not int or not 0<=v<config.vocab_size for v in eos):raise ValueError('explicit original EOS identity required')
    context=list(ids);generated=[];devices=[]if runtime.device.type=='cpu'else[runtime.device.index]
    with torch.random.fork_rng(devices=devices),native.network.scaled(strength):
        torch.manual_seed(seed)
        for unused in range(max_new_tokens):
            require_owner(str(runtime.device));hidden=runtime.hidden(context)[:,-1]
            # Exactly the original complete BF16 head. No logit coordinate subset,
            # pooling, projected vocabulary, sampling defaults or hidden decoder.
            weight=runtime.store.read('lm_head.weight',runtime.device)
            logits=F.linear(hidden,weight).float();del weight
            if logits.shape!=(1,config.vocab_size)or not torch.isfinite(logits).all():raise ValueError('complete finite native vocabulary logits required')
            token=int(logits.argmax(-1).item());generated.append(token);context.append(token)
            if token in eos:break
    text=tokenizer.decode(generated,skip_special_tokens=False)
    return TextOutput(text,dict(schema='Ripple095-native-finished-text-v1',model=runtime.identity,selection='raw',weight_EMA=None,
        prompt_token_ids=list(ids),new_token_ids=generated,eos_token_ids=list(eos),native_greedy_argmax=True,
        complete_vocabulary=config.vocab_size,full80_layer_hidden=True,KV_cache=False,context_truncated=False,
        strength=strength,seed=seed,actual_token_budget=max_new_tokens,quality_or_LR_qualified=False))
