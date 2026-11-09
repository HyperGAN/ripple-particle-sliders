"""Import-free-of-Torch audit of the proposed native width-5120 source cohort.

This only reads source/metadata/archive bytes. It never installs, imports a
training engine, selects hardware, returns runtime admission, or edits a profile.
"""
import argparse
import email.parser
import hashlib
import io
from importlib import metadata
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).parent
PROFILE = 'source-801ae1e-continuous-v1'
COMMIT = '801ae1ead8f88b88eca8f5398137e3e1bc01ad36'
WHEEL_SHA256 = '263f8edead2f972b87d03a93780e95929c8c03d2989f3770313b88cb3699c9d0'
WHEEL_BYTES = 296898
CORE_COMMIT = '5d68473a098801ee1ab3ebbb12463daef51c6346'


class ProfileError(ValueError):
    pass


def proposal():
    value = json.loads((ROOT/'profiles/continuous5120.json').read_text())
    if (value.get('pg_source_profile'), value.get('pg_source_commit'),
            value.get('pg_wheel_sha256'), value.get('core_commit')) != (
            PROFILE, COMMIT, WHEEL_SHA256, CORE_COMMIT):
        raise ProfileError('proposal contradicts immutable shared source identity')
    validate_options(value)
    if (value.get('schema')!='ripple-continuous5120-source-proposal-v1' or
            value.get('status')!='UNQUALIFIED_PRODUCT_PROPOSAL' or
            any(value.get(key) is not False for key in
                ('training_enabled','runtime_qualified','gpu_admitted','quality_qualified'))):
        raise ProfileError('metadata proposal cannot enable training or claim runtime qualification')
    return value


def validate_options(value):
    """A paired declaration with original native geometry; never width heuristics."""
    if value.get('pg_version') != '0.9.4':
        raise ProfileError('exact ParticleGAN 0.9.4 source cohort required')
    if (value.get('pg_source_profile'), value.get('penalty_input_domain')) != (PROFILE, 'continuous'):
        raise ProfileError('explicit exact-source/continuous pair required; no release alias')
    shape = value.get('response_shape')
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(v) is not int for v in shape) or tuple(shape) != (1,5120)
            or value.get('response_domain') != 'continuous_vector'):
        raise ProfileError('original [1,5120] continuous vector required without projection or reshape')
    if value.get('cohort') != 'ripple-nvfp4-pg094-source801-continuous5120-v1-candidate':
        raise ProfileError('distinct explicit candidate cohort required')
    if type(value.get('condition_width')) is not int or value['condition_width'] != 5124:
        raise ProfileError('native conditioning width must remain 5124')
    return dict(pg_source_profile=PROFILE, penalty_input_domain='continuous', pg_version='0.9.4')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(1024*1024): result.update(chunk)
    return result.hexdigest()


def python_inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*.py')) if p.is_file()}


def compare_inventories(actual, expected):
    return dict(missing=sorted(expected.keys()-actual.keys()),
                extra=sorted(actual.keys()-expected.keys()),
                changed=sorted(k for k in actual.keys() & expected.keys() if actual[k]!=expected[k]))


def equal_inventory(actual, expected):
    return not any(compare_inventories(actual, expected).values())


def audit_wheel(path, manifest):
    # Audit a bounded immutable byte snapshot, not a second pathname read after
    # hashing; an unrelated large file must also fail without unbounded I/O.
    with Path(path).open('rb') as file:
        data = file.read(WHEEL_BYTES+1)
    if len(data)!=WHEEL_BYTES or hashlib.sha256(data).hexdigest()!=WHEEL_SHA256:
        return dict(exact_archive=False, source_closure_match=False,
                    qualified_for_runtime=False, reason='candidate wheel hash differs')
    files, version = {}, None
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            name = entry.filename
            if entry.file_size > 5*1024**2:
                raise ProfileError('unexpectedly large candidate source entry')
            if name.startswith('particlegan/') and name.endswith('.py'):
                relative = name[len('particlegan/'):]
                if relative in files or '..' in Path(relative).parts:
                    raise ProfileError('duplicate or invalid candidate source path')
                files[relative] = hashlib.sha256(archive.read(entry)).hexdigest()
            elif name.endswith('.dist-info/METADATA'):
                document = email.parser.BytesParser().parsebytes(archive.read(entry))
                if document.get('Name','').lower() == 'particlegan': version = document['Version']
    match = equal_inventory(files, manifest['python_sources'])
    return dict(exact_archive=True, source_closure_match=match, metadata_version=version,
                source_file_count=len(files), differences=compare_inventories(files, manifest['python_sources']),
                qualified_for_runtime=False)


def audit_files(core_root, pg_root, *, core_version, core_origin, pg_version, candidate_wheel=None):
    selected = proposal()
    core_root, pg_root = Path(core_root), Path(pg_root)
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    expected = lock['core_sources']
    actual = {}
    for package, folder in [('particle_sliders',core_root),
                            ('concept_slider_core',core_root.parent/'concept_slider_core')]:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                actual[package+'/'+str(p.relative_to(folder))] = digest(p)
    core_match = equal_inventory(actual, expected)
    if not isinstance(core_origin, dict): core_origin = {}
    vcs = core_origin.get('vcs_info', {})
    if not isinstance(vcs, dict): vcs = {}
    directory = core_origin.get('dir_info', {})
    if not isinstance(directory, dict): directory = {'editable':True}
    origin_match = (core_version=='0.12.0' and core_origin.get('url')==
        'https://github.com/HyperGAN/particle-sliders.git' and
        core_origin.get('subdirectory')=='packages/particle-sliders-core' and
        vcs.get('vcs')=='git' and vcs.get('commit_id')==CORE_COMMIT and
        not directory.get('editable',False) and 'archive_info' not in core_origin)
    # Refuse an altered core before trusting its candidate source manifest.
    if not core_match or not origin_match or lock.get('core_commit')!=CORE_COMMIT:
        return dict(status='REFUSED_CORE_IDENTITY', core_sources_match=core_match,
            core_origin_match=origin_match, differences=compare_inventories(actual, expected),
            runtime_admitted=False, training_enabled=False)
    manifest_path = core_root/'pg094_continuous.json'
    manifest = json.loads(manifest_path.read_text())
    if (digest(manifest_path)!=selected['core_profile_manifest_sha256'] or
            (manifest.get('version'),manifest.get('source_profile'),manifest.get('source_commit'),
             manifest.get('wheel_sha256')) != ('0.9.4',PROFILE,COMMIT,WHEEL_SHA256)):
        raise ProfileError('shared continuous manifest contradicts candidate identity')
    release = json.loads((core_root/'pg094_release.json').read_text())
    sources = python_inventory(pg_root)
    source_match = equal_inventory(sources, manifest['python_sources'])
    installed_matches = source_match and pg_version=='0.9.4'
    artifact = audit_wheel(candidate_wheel, manifest) if candidate_wheel else None
    artifact_rejected = artifact is not None and not (
        artifact.get('exact_archive') and artifact.get('source_closure_match')
        and artifact.get('metadata_version')=='0.9.4')
    status = ('REFUSED_CANDIDATE_ARTIFACT' if artifact_rejected else
              'SOURCE_CHECKS_PASS_RUNTIME_PENDING' if installed_matches else 'HOLD_INSTALLED_SOURCE_MISMATCH')
    return dict(status=status,
        proposed_cohort=selected['cohort'], response_shape=[1,5120],
        core_sources_match=True, core_origin_match=True, pg_metadata_version=pg_version,
        installed_matches_candidate=installed_matches,
        installed_matches_released_source=equal_inventory(sources, release['python_sources']),
        candidate_profile=PROFILE, penalty_input_domain='continuous',
        source_file_count=len(sources), differences=compare_inventories(sources, manifest['python_sources']),
        candidate_wheel=artifact, runtime_admitted=False, training_enabled=False,
        imported_or_executed_engine=False, hardware_selected=False,
        qualification='source/metadata inspection only; shared runtime gate and product tests remain required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-wheel', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    core, pg = metadata.distribution('particle-sliders-core'), metadata.distribution('particlegan')
    report = audit_files(core.locate_file('particle_sliders'), pg.locate_file('particlegan'),
        core_version=core.version, core_origin=json.loads(core.read_text('direct_url.json') or '{}'),
        pg_version=pg.version, candidate_wheel=args.candidate_wheel)
    encoded = json.dumps(report,indent=2)+'\n'
    if args.out: args.out.write_text(encoded)
    print(encoded)
    if report['status'] != 'SOURCE_CHECKS_PASS_RUNTIME_PENDING':
        raise SystemExit(2)


if __name__=='__main__': main()
