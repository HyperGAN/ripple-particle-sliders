"""Complete original packed cache; no BF16 checkpoint/backend or reduced response."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import torch
from torch import nn
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter
from particle_sliders.restoration import _restore_all
from .contracts import MODEL,REVISION,DEFAULTS,file_hash
from .ownership import require_owner
from .streaming import FileTensorStore,StreamingModel
from .vendor.configuration_ricursive import RicursiveConfig
from .adapter import Slider

class NativeRuntime:
    def __init__(self,options):
        self.closed=False;self.store=None;self.model=None;self.tokenizer=None
        self.device=torch.device(options['device']);require_owner(str(self.device))
        self.trainable_device=self.device
        root=Path(options['cache_dir']);raw=(root/'manifest.json').read_bytes()
        if len(raw)>8*2**20 or hashlib.sha256(raw).hexdigest()!=options['cache_manifest_sha256']:raise ValueError('exact complete original cache manifest required')
        m=json.loads(raw);official=json.loads(Path(__file__).with_name('vendor').joinpath('config.json').read_text())
        if(m.get('schema')!='ripple-raw-tensor-cache-v1'or m.get('source',{}).get('model')!=MODEL
            or m['source'].get('revision')!=REVISION or m.get('config')!=official):raise ValueError('original full Ripple revision/config required')
        if(official['hidden_size'],official['intermediate_size'],official['num_hidden_layers'])!=(5120,40960,80):raise ValueError('complete original native geometry required')
        try:
            self.store=FileTensorStore(root)
            self.model=StreamingModel(RicursiveConfig.from_dict(official),self.store,device=self.device,
                chunk_rows=DEFAULTS['chunk_rows'],activation=DEFAULTS['activation']).eval().requires_grad_(False)
            expected={'model.embed_tokens.weight','model.norm.weight','lm_head.weight'}
            for layer in self.model.layers:
                expected.update('model.layers.'+str(layer.index)+'.'+n for n in layer.dense_names)
                expected.update('model.layers.'+str(layer.index)+'.mlp.'+leaf+'_proj.'+suffix
                    for leaf in('gate','up','down')for suffix in('weight','weight_scale','weight_scale_2','input_scale'))
            if set(self.store.entries)!=expected:raise ValueError('ALL80 original blocks/embedding/norm/full LMhead required; block-only cache is not a full model')
            vocab=official['vocab_size']
            for name,shape in(('model.embed_tokens.weight',(vocab,5120)),('lm_head.weight',(vocab,5120)),('model.norm.weight',(5120,))):
                d=self.store.entries[name]
                if d['dtype']!='BF16'or d['shape']!=shape:raise ValueError('complete original BF16 embedding/norm/head geometry required')
            self.identity=dict(model=MODEL,revision=REVISION,cache_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                numerical_profile='Original NVFP4/ModelOpt activation identitySTE; chunk256 BF16 reference GEMM, not native FP4 TensorCore',
                full80_layer_config=official,selection='raw',weight_EMA=None,
                native_product_sources={str(p.relative_to(Path(__file__).parent)):file_hash(p)for p in sorted(Path(__file__).parent.rglob('*'))if p.is_file()and p.suffix in('.py','.json','.hndl')})
        except BaseException as primary:_restore_all([self.close],primary=primary);raise
    def hidden(self,ids):
        require_owner(str(self.device))
        return self.model.hidden(list(ids))
    def load_tokenizer(self):
        if self.tokenizer is not None:return self.tokenizer
        # Tokenizer Source identity is part of the same authenticated manifest.
        spec=self.store.manifest.get('tokenizer')
        if not isinstance(spec,dict)or spec.get('model')!=MODEL or spec.get('revision')!=REVISION or not isinstance(spec.get('files'),dict)or not spec['files']:raise ValueError('original checkpoint tokenizer Source/pins required for finished text')
        root=self.store.root/'tokenizer'
        for name,row in spec['files'].items():
            p=root/name
            if not p.resolve().is_relative_to(root.resolve())or p.is_symlink()or p.stat().st_size!=row['bytes']or file_hash(p)!=row['sha256']:raise ValueError('exact original tokenizer file required:'+name)
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        if len(self.tokenizer)>self.model.config.vocab_size:raise ValueError('tokenizer vocabulary exceeds original native head')
        return self.tokenizer
    def close(self):
        if self.store is not None:self.store.close()
        self.model=None;self.tokenizer=None;self.closed=True

class _Host(nn.Module):
    def __init__(self,model):super().__init__();self.model=model;self.config=model.config

class NativeResponse(ARLastHiddenResponseAdapter):
    family='ripple095_full_native_last_hidden5120'
    def __init__(self,runtime):
        self.runtime=runtime;self.slider=Slider(runtime.model,device=runtime.device)
        self.fresh_initialization=None;self.teacher_cache=None;self._native_closed=False
        try:
            super().__init__(_Host(runtime.model),self.slider,model_identity=deepcopy(runtime.identity),
                checkpoint_modules=tuple(runtime.model.layers),pg_source_profile='release-0.9.5',penalty_input_domain='continuous')
        except BaseException as primary:_restore_all([self.slider.close,runtime.close],primary=primary);raise
    def _hidden(self,ids):return self.runtime.hidden(ids)[:,-1]
    def response(self,*,ids,strength=1.,seed=0):
        del seed
        with torch.no_grad():return self._response(tuple(ids),scale=strength).detach().clone()
    def metadata(self):
        from .contracts import SOURCE_PROFILE,INITIALIZATION_POLICY,WEIGHT_POLICY
        return dict(super().metadata(),source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,
            weight_policy=WEIGHT_POLICY,fresh_initialization=self.fresh_initialization,teacher_cache=self.teacher_cache,
            native_full_width=5120,numerical_profile=self.runtime.identity['numerical_profile'])
    def close(self):
        if self._native_closed:return
        _restore_all([self.close_checkpointing,self.slider.close,self.runtime.close]);self.model.model=None;self._native_closed=True
