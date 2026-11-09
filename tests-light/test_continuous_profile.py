"""Standard-library-only metadata tests; no engine, Tensor, model or CUDA import."""
from copy import deepcopy
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ripple_profile_audit',
    Path(os.environ.get('RIPPLE_PROFILE_MODULE',
         ROOT/'src/ripple_particle_sliders/continuous_profile.py')))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class DeclarationTests(unittest.TestCase):
    def test_exact_candidate_preserves_geometry_and_pair(self):
        plan = profile.proposal()
        options = profile.validate_options(plan)
        self.assertEqual(plan['response_shape'], [1,5120])
        self.assertEqual(options, dict(pg_version='0.9.4', pg_source_profile=profile.PROFILE,
                                       penalty_input_domain='continuous'))
        self.assertFalse(plan['training_enabled'])
        self.assertNotIn('torch', sys.modules)

    def test_cross_pairs_release_alias_and_unknown_profile_refuse(self):
        for update in [dict(pg_source_profile='release-0.9.4'),
                       dict(penalty_input_domain='auto'), dict(penalty_input_domain='categorical'),
                       dict(pg_source_profile='source-801ae1e'), dict(pg_version='0.9.4+continuous')]:
            with self.subTest(update=update), self.assertRaises(profile.ProfileError):
                profile.validate_options(dict(profile.proposal(), **update))

    def test_native_shape_cannot_be_reduced_reinterpreted_or_broadcast(self):
        for shape in ([1,169], [2,5120], [1,5120,1,1], [True,5120], ['1',5120], None):
            with self.subTest(shape=shape), self.assertRaises(profile.ProfileError):
                profile.validate_options(dict(profile.proposal(), response_shape=shape))
        with self.assertRaises(profile.ProfileError):
            profile.validate_options(dict(profile.proposal(), response_domain='continuous_field'))

    def test_old_cohort_and_changed_condition_width_refuse(self):
        for update in [dict(cohort='ripple-pg094-hypergan-hndl-v1-cpu-fixture'),
                       dict(condition_width=5123), dict(condition_width=5124.0)]:
            with self.subTest(update=update), self.assertRaises(profile.ProfileError):
                profile.validate_options(dict(profile.proposal(), **update))

    def test_inventory_requires_every_source_and_no_overlay(self):
        expected = {'a.py':'1', 'b.py':'2'}
        self.assertTrue(profile.equal_inventory(dict(expected), expected))
        self.assertFalse(profile.equal_inventory({'a.py':'1'}, expected))
        self.assertFalse(profile.equal_inventory(dict(expected, **{'overlay.py':'x'}), expected))
        self.assertFalse(profile.equal_inventory({'a.py':'1', 'b.py':'changed'}, expected))

    def test_metadata_cannot_turn_into_admission_or_quality_claim(self):
        original=profile.proposal()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'profiles').mkdir()
            for key,value in [('training_enabled',True),('runtime_qualified',True),
                              ('gpu_admitted',True),('quality_qualified',True),
                              ('training_enabled',0),('status','TRAINING_READY')]:
                candidate=dict(original,**{key:value})
                (root/'profiles/continuous5120.json').write_text(json.dumps(candidate))
                with self.subTest(key=key,value=value), patch.object(profile,'ROOT',root), self.assertRaises(profile.ProfileError):
                    profile.proposal()


@unittest.skipUnless(os.environ.get('RIPPLE_CORE_ROOT') and os.environ.get('RIPPLE_PG_ROOT')
                     and os.environ.get('RIPPLE_CONTINUOUS_WHEEL'), 'explicit read-only source paths required')
class SourceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = Path(os.environ['RIPPLE_CORE_ROOT'])
        cls.pg = Path(os.environ['RIPPLE_PG_ROOT'])
        cls.wheel = Path(os.environ['RIPPLE_CONTINUOUS_WHEEL'])
        if profile.digest(cls.wheel) != profile.WHEEL_SHA256:
            raise ValueError('metadata fixture requires the exact known source archive')
        cls.manifest = json.loads((cls.core/'pg094_continuous.json').read_text())
        cls.origin = dict(url='https://github.com/HyperGAN/particle-sliders.git',
            subdirectory='packages/particle-sliders-core',
            vcs_info=dict(vcs='git',commit_id=profile.CORE_COMMIT))
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.snapshot = Path(cls.temporary.name)/'source-snapshot'
        # This is an unimported metadata fixture from the exact archive; never
        # install it, put it on sys.path, or present it as the active runtime.
        with zipfile.ZipFile(cls.wheel) as archive:
            for name in archive.namelist():
                if name.startswith('particlegan/') and name.endswith('.py'):
                    relative = Path(name[len('particlegan/'):])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('invalid fixture source path')
                    target = cls.snapshot/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.read(name))

    def audit(self, pg=None, **kwargs):
        values = dict(core_version='0.12.0', core_origin=self.origin, pg_version='0.9.4',
                      candidate_wheel=self.wheel)
        values.update(kwargs)
        return profile.audit_files(self.core, pg or self.pg, **values)

    def test_exact_existing_archive_matches_shared_manifest(self):
        result = profile.audit_wheel(self.wheel,self.manifest)
        self.assertTrue(result['exact_archive'])
        self.assertTrue(result['source_closure_match'])
        self.assertEqual(result['metadata_version'],'0.9.4')
        self.assertFalse(result['qualified_for_runtime'])

    def test_installed_release_is_not_the_continuous_candidate(self):
        result = self.audit()
        self.assertTrue(result['core_sources_match'])
        self.assertTrue(result['installed_matches_released_source'])
        self.assertFalse(result['installed_matches_candidate'])
        self.assertEqual(result['differences']['changed'],['bispf.py','recipes.py'])
        self.assertFalse(result['runtime_admitted'])

    def test_exact_snapshot_passes_only_source_checks(self):
        result = self.audit(self.snapshot)
        self.assertTrue(result['installed_matches_candidate'])
        self.assertEqual(result['status'],'SOURCE_CHECKS_PASS_RUNTIME_PENDING')
        self.assertFalse(result['training_enabled'])
        self.assertFalse(result['runtime_admitted'])
        self.assertFalse(result['imported_or_executed_engine'])
        self.assertNotIn('torch',sys.modules)

    def test_resolved_bispf_fields_match_without_engine_import(self):
        modules = []
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            for label, path in [('released',self.pg/'recipes.py'),
                                ('source801',self.snapshot/'recipes.py')]:
                name = '_ripple_recipe_metadata_'+label
                spec = importlib.util.spec_from_file_location(name,path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                modules.append(name)
                spec.loader.exec_module(module)
            released, candidate = [sys.modules[name] for name in modules]
            for preset in ('bispf_optimized','bispf'):
                options = dict(batch_size=2,total_steps=4,num_particles=128,z_dim=4)
                self.assertEqual(released.get_recipe_fields(preset,**options),
                                 candidate.get_recipe_fields(preset,**options))
            self.assertNotIn('torch',sys.modules)
        finally:
            for name in modules: sys.modules.pop(name,None)
            sys.dont_write_bytecode = previous_bytecode

    def test_wrong_core_origin_version_or_editable_source_refuses(self):
        for origin in [dict(self.origin, vcs_info=dict(vcs='git',commit_id='0'*40)),
                       dict(self.origin, dir_info=dict(editable=True)),
                       dict(self.origin, archive_info={}), None]:
            with self.subTest(origin=origin):
                self.assertEqual(self.audit(core_origin=origin)['status'],'REFUSED_CORE_IDENTITY')
        self.assertEqual(self.audit(core_version='0.11.0')['status'],'REFUSED_CORE_IDENTITY')
        self.assertFalse(self.audit(self.snapshot,pg_version='0.9.5')['installed_matches_candidate'])

    def test_changed_archive_cannot_claim_candidate_identity(self):
        path = Path(self.temporary.name)/'changed.whl'
        data = bytearray(self.wheel.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        result = profile.audit_wheel(path,self.manifest)
        self.assertFalse(result['exact_archive'])
        self.assertFalse(result['source_closure_match'])
        self.assertEqual(self.audit(self.snapshot,candidate_wheel=path)['status'],
                         'REFUSED_CANDIDATE_ARTIFACT')

    def test_cli_hold_is_nonzero_and_positive_source_audit_still_cannot_admit(self):
        class Distribution:
            def __init__(self, root, version): self.root,self.version=root,version
            def locate_file(self,name): return self.root
            def read_text(self,name): return json.dumps(SourceArtifactTests.origin)
        old = sys.argv
        try:
            for pg_root, code in [(self.pg,2),(self.snapshot,0)]:
                distributions={'particle-sliders-core':Distribution(self.core,'0.12.0'),
                               'particlegan':Distribution(pg_root,'0.9.4')}
                output=io.StringIO()
                sys.argv=['ripple-continuous-profile-audit','--candidate-wheel',str(self.wheel)]
                with patch.object(profile.metadata,'distribution',side_effect=distributions.__getitem__), contextlib.redirect_stdout(output):
                    if code:
                        with self.assertRaises(SystemExit) as raised: profile.main()
                        self.assertEqual(raised.exception.code,code)
                    else: profile.main()
                report=json.loads(output.getvalue())
                self.assertFalse(report['runtime_admitted'])
                self.assertFalse(report['training_enabled'])
        finally: sys.argv=old


if __name__=='__main__': unittest.main()
