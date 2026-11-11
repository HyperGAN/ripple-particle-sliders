"""New native5120 cohort; old094 packets/fixtures keep their original namespace."""
from pathlib import Path
import hashlib,json,math
CORE_REVISION='41f6066f2d2644902670d5eb736de052e1a70553'
BACKEND='ripple_native95'
SOURCE_PROFILE='ripple-native5120-NVFP4-STE-continuous-pg095-hndl-v1'
INITIALIZATION_POLICY='ripple-host-RMS-QR-zero-up-bank-R2-std1-v1'
PROTOCOL='ripple095-native-last-hidden5120-fullvocab-finished-text-v1'
WEIGHT_POLICY='raw-no-weight-EMA'
MODEL='ricursiveAI/Ripple-65B-Preview-NVFP4'
REVISION='99eea4477a001a41b86914e41feb4b2ed86105e5'
DEFAULTS=dict(rank=8,alpha=8.,particles=128,particle_dim=4,response_width=5120,chunk_rows=256,activation='w4a4_ste')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def architecture(options):return {k:options.get(k,v)for k,v in DEFAULTS.items()}
def rate_overrides(options):
    values={k:options[k]for k in('lr','d_lr_mult','prior_lr_mult')if k in options}
    if any(type(v)not in(int,float)or not math.isfinite(v)or v<=0 for v in values.values()):raise ValueError('finite positive non-Boolean common rates required')
    return values

def validate_profile(options):
    if 'ema_decay'not in options or options['ema_decay']is not None or options.get('ema')is not None:raise ValueError('explicit ema_decay=None; all weight EMA refused before allocation')
    if(options.get('source_profile'),options.get('initialization_policy'),options.get('pg_version'),options.get('pg_source_profile'),options.get('penalty_input_domain'))!=(SOURCE_PROFILE,INITIALIZATION_POLICY,'0.9.5','release-0.9.5','continuous'):
        raise ValueError('explicit new095/native5120/continuous initialization profile required')
    rate_overrides(options)

def validate_options(options,*,training=False):
    allowed=set(DEFAULTS)|{'cache_dir','cache_manifest_sha256','device'}
    if training:allowed|={'training_file','teacher_cache','steps','batch','seed','ema_decay','source_profile','initialization_policy','pg_version','pg_source_profile','penalty_input_domain','lr','d_lr_mult','prior_lr_mult'}
    if training:validate_profile(options)
    if set(options)-allowed:raise ValueError('unknown Ripple095 option; no projection/precision/backend fallback')
    a=architecture(options)
    if a!=DEFAULTS or any(type(a[k])is not int for k in('rank','particles','particle_dim','response_width','chunk_rows')):raise ValueError('preserve full original architecture/rank8/128x4/chunk256/W4A4-STE')
    if options.get('device')not in('cpu','cuda:0','cuda:1'):raise ValueError('explicit single execution device required')
    from .ownership import require_owner
    require_owner(options['device'])
    if not isinstance(options.get('cache_manifest_sha256'),str)or len(options['cache_manifest_sha256'])!=64 or any(c not in'0123456789abcdef'for c in options['cache_manifest_sha256']):raise ValueError('exact authenticated full original cache manifest SHA required')
    root=Path(options.get('cache_dir',''))
    if not isinstance(options.get('cache_dir'),str)or not root.is_absolute()or not root.is_dir():raise ValueError('explicit offline full original cache_dir required')
    if training:
        for k,default,low,high in(('steps',32,1,2**31-1),('batch',2,2,512),('seed',7,0,2**63-100)):
            v=options.get(k,default)
            if type(v)is not int or not low<=v<=high:raise ValueError('invalid '+k)
        if not isinstance(options.get('training_file'),str)or not Path(options['training_file']).is_file():raise ValueError('matched local complete token rows required')
        if not isinstance(options.get('teacher_cache'),str)or not Path(options['teacher_cache']).is_absolute():raise ValueError('explicit durable teacher cache required')

def validate_installed_core():
    from importlib import metadata
    lock=json.loads(Path(__file__).with_name('current_core.json').read_text());d=metadata.distribution('particle-sliders-core')
    if d.version!=lock['version']or lock['commit']!=CORE_REVISION:raise ValueError('exact current core0.14.1 source required')
    for name,sha in lock['checked_api_files'].items():
        if file_hash(d.locate_file('particle_sliders/'+name))!=sha:raise ValueError('current core Source differs:'+name)
    if metadata.version('transformers')!='4.57.6'or metadata.version('particlegan')!='0.9.5':raise ValueError('explicit new TF4.57.6/PG095 profile required; no old qualification transfer')
