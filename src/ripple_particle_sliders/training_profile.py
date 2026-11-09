"""Allocation-free discovery for the shared particle-slider command."""
TRAINING_PROFILE = dict(
    schema='particle-slider-profile-v1', name='ripple-cpu-fixture',
    product='ripple-particle-sliders', config_kind='session', status='fixture',
    interface='response', backend='ripple-particle-sliders', runtime={},
    capabilities=dict(resume=True, periodic_checkpoint=True, check=True),
    default_checkpoint_every=2,
    qualification='Random FP32 CPU Ripple architecture fixture; native NVFP4 training blocked')
