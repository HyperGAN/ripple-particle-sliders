"""Execution-owned factory derivative: only individual teacher durability changes.

Every native class, initializer, critic and official PG formula comes from exact
777f43ce. Original inference specification/RAW factory Source stays unchanged.
The new training prepare Source is honestly captured in its own FULL manifest.
"""
from copy import deepcopy
from pathlib import Path
import torch
from ripple_native95.factory import (validate_training,load_runtime,NativeResponse,Critic,
    initialize_owned,engine_options,ConditionalPG095,INFERENCE,PROTOCOL,architecture,SOURCE_PROFILE,
    INITIALIZATION_POLICY,WEIGHT_POLICY,rate_overrides,digest,Construction,Binding,_restore_all,
    preflight,CAP,BACKEND,ResponseBackend,construction_for,read_packet)
from durable_teachers import teachers as durable_teachers

def prepare_training(options,*,construction=None,check):
    validate_training(options)
    if not isinstance(construction,Construction)or construction.options_sha256!=digest(options):raise ValueError('explicit fresh/resume intent required')
    device=torch.device(options['device']);devices=[]if device.type=='cpu'else[device.index]
    runtime=native=None
    try:
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get('seed',7),devices)
            runtime=load_runtime(options);native=NativeResponse(runtime)
        check()
        rows,conditions=durable_teachers(native,options,construction,check)
        check()
        if conditions.shape!=(len(rows),5124):raise ValueError('full frozen native5120+four literal condition coordinates required')
        with torch.random.fork_rng(devices=devices):
            seed_initialization_rng(options.get('seed',7),devices)
            critic=Critic(5120,5124).float().to(device)
            receipt=initialize_owned(native.slider,critic,construction,options)
            native.fresh_initialization=receipt.as_dict()
        if any(p.device!=device or p.dtype!=torch.float32 for p in native.network.parameters()):raise ValueError('all original trainable Parameters FP32 on one explicit PGdevice required')
        game=ConditionalPG095(native,rows,critic,conditions=conditions,particles=native.network.particles,
            response_domain='continuous_vector',total_steps=options.get('steps',32),batch_size=options.get('batch',2),
            seed=options.get('seed',7),recipe_overrides=rate_overrides(options),**engine_options(options))
        spec=INFERENCE.specification(native,model_identity=native.model_identity,protocol=PROTOCOL,
            configuration=dict(architecture=architecture(options),source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,
                weight_policy=WEIGHT_POLICY,initialization=receipt.as_dict()))
        def response(*,index,split,strength,seed):
            del seed
            if split!='train'or type(index)is not int or not 0<=index<len(rows):raise ValueError('exact prepared train row required')
            with torch.no_grad():return native._response(rows[index].row.neutral_ids,scale=strength).detach().clone()
        return Binding(game,native.close,inference=spec,response=response)
    except BaseException as primary:_restore_all([native.close]if native is not None else[runtime.close]if runtime is not None else[],primary=primary);raise


def open_training(config,*,check,resume=None):
    validate_training(config.options)
    packet=deepcopy(resume)if isinstance(resume,dict)else read_packet(resume)if resume is not None else None
    construction=construction_for(config,packet)
    def prepare(options,*,construction):return prepare_training(options,construction=construction,check=check)
    backend=ResponseBackend(BACKEND,CAP,prepare,validate_training,preflight=preflight)
    if packet is not None and not Path(config.options['teacher_cache']).is_file():raise ValueError('exact original teacher cache required; no replay')
    session=backend.open(config,operation='resume'if packet is not None else'train',construction=construction)
    try:
        if packet is not None:session.restore(packet)
        return session
    except BaseException as primary:_restore_all([session.close],primary=primary);raise
