"""Exact original attention-only HNDL branches, unit-std bank and Linear16 critic."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from particle_sliders.fibo_hndl import HNDLParticleProjection,verified_builder
from particle_sliders.restoration import _restore_all
from .vendor.modeling_ricursive import RicursiveAttention
CRITIC_GRAPH='''joined = concat(response, condition, axis=1)
features = linear(joined, 16, name="hidden")
features = silu(features)
linear(features, 1, name="score")
'''

def targets(model):
    result={};seen=set()
    for name,module in model.named_modules():
        if not isinstance(module,RicursiveAttention):continue
        for leaf in('q_proj','k_proj','v_proj','o_proj'):
            child=getattr(module,leaf)
            if type(child)is not nn.Linear or id(child)in seen or getattr(child.forward,'__func__',None)is not nn.Linear.forward:raise ValueError('unique original unwrapped attention Linear required')
            result[name+'.'+leaf]=child;seen.add(id(child))
    if len(result)!=320:raise ValueError('all80 original Q/K/V/O attention branches required; no MLP/gate adaptation')
    return result

class Slider(nn.Module):
    def __init__(self,model,*,device):
        super().__init__();native=targets(model);builder,identity=verified_builder()
        self.particles=nn.Parameter(torch.randn(128,4,device=device,dtype=torch.float32))
        self.adapters=nn.ModuleDict();self._hosts=[];self.closed=False
        # Construct every graph before mutating any original forward.
        for n,(name,module)in enumerate(native.items()):
            a=HNDLParticleProjection(name,module,self.particles,8,8,builder=builder,identity=identity,attach=False)
            self.adapters[str(n)]=a.float().to(device)
        try:
            for module,a in zip(native.values(),self.adapters.values()):
                self._hosts.append((module,module.forward,a));module.forward=a.forward
        except BaseException as primary:_restore_all([self.close],primary=primary);raise
    @contextmanager
    def scaled(self,strength):
        if self.closed:raise RuntimeError('closed slider')
        if type(strength)not in(int,float)or not math.isfinite(strength):raise ValueError('finite scalar strength required')
        previous=[a.multiplier for a in self.adapters.values()]
        try:
            for a in self.adapters.values():a.multiplier=float(strength)
            yield self
        finally:
            for a,value in zip(self.adapters.values(),previous):a.multiplier=value
    def close(self):
        errors=[]
        for module,original,a in reversed(self._hosts):
            a.multiplier=0.
            try:
                if module.forward!=original:
                    if module.forward!=a.forward:raise RuntimeError('foreign native forward retained; original ownership lost')
                    module.forward=original
            except BaseException as error:errors.append(error)
        if errors:raise errors[0]
        self.closed=True

class Critic(nn.Module):
    score_bound=0
    def __init__(self,width,condition_width):
        super().__init__();builder,self.identity=verified_builder()
        if(width,condition_width)!=(5120,5124):raise ValueError('full original response5120/condition5124 required')
        self.graph=builder(CRITIC_GRAPH,input_shape={'response':['B',width],'condition':['B',condition_width]},output_shape=['B',1])
    def forward(self,response,condition):return self.graph(response=response,condition=condition).squeeze(-1)
