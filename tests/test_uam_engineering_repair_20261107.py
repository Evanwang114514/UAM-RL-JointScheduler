# -*- coding: utf-8 -*-
"""标准库回归检查；深度学习库使用替身，不训练、不导入真实torch、不使用GPU。"""
import ast
import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
spec = importlib.util.spec_from_file_location('uam_engineering_repair_tested', PROJECT / 'repair_uam_dima_ppo_20261107_ONEFILE.py')
repair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = repair
spec.loader.exec_module(repair)


class FakeNorm:
    def __init__(self):
        self.training = True
        self.norm_reward = True

    def normalize_obs(self, value):
        assert self.training is False and self.norm_reward is False
        return value


class FakeArray:
    def __getitem__(self, index):
        return self


class FakeTensor:
    def __init__(self, values, dtype):
        self.values, self.dtype = list(values), dtype

    def to(self, other):
        return FakeTensor(self.values, other.dtype)

    def __setitem__(self, index, value):
        if self.dtype != value.dtype:
            raise RuntimeError('dtype mismatch')
        self.values = value.values


class Repairs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='uam_repair_test_', dir=WORKSPACE / 'scratch')
        self.project = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        repair.write_json(path, value)

    def test_locked_entries_and_plan_unchanged(self):
        for suite, (name, sha) in repair.ENTRY.items():
            self.assertEqual(repair.digest(PROJECT / name), sha)
            shutil.copy2(PROJECT / name, self.project / name)
            root = self.project / 'serial_runs' / ('uam_' + suite + '_author_20261106_v1')
            root.mkdir(parents=True)
            entry_spec = importlib.util.spec_from_file_location('contract_' + suite, self.project / name)
            entry = importlib.util.module_from_spec(entry_spec)
            entry_spec.loader.exec_module(entry)
            plan = entry.plan_make()
            plan['entrypoint_sha256'] = sha
            self.write(root / 'PLAN.json', plan)
            _, _, result = repair.load_entry(self.project, suite)
            self.assertEqual(result, plan)
            self.assertLessEqual(result['planned_real_steps'], result['budget_cap'])

    def test_path_escape_and_symlink_fail_closed(self):
        with self.assertRaises((ValueError, RuntimeError)):
            repair.checked(self.project / '..' / 'outside', self.project, exists=False)
        with self.assertRaises((ValueError, RuntimeError)):
            repair.checked(self.project.parent / 'outside', self.project, exists=False)

    def test_dtype_cast_preserves_binary_mask(self):
        old = 'def sample():\n' + '    if True:\n' + '        if True:\n' + '            if True:\n' + repair.DIMA_OLD + '\n'
        fixed = repair.dima_text_patch(old)
        ast.parse(fixed)
        self.assertEqual(fixed.count("av_action.to(info['av_action'])"), 1)
        destination = FakeTensor([0, 0, 0], 'long')
        source = FakeTensor([1., 0., 1.], 'float')
        with self.assertRaises(RuntimeError):
            destination[True] = source
        exec(textwrap.dedent(repair.DIMA_NEW), {'info': {'av_action': destination}, 'av_action': source, 'dead': True})
        self.assertEqual(destination.values, [1., 0., 1.])
        self.assertEqual(destination.dtype, 'long')
        with self.assertRaises(RuntimeError):
            repair.dima_text_patch(fixed)

    def test_dima_real_author_patch_and_manifest_idempotence(self):
        name = repair.ENTRY['dima'][0]
        entry_spec = importlib.util.spec_from_file_location('repair_dima_fixture', PROJECT / name)
        entry = importlib.util.module_from_spec(entry_spec)
        entry_spec.loader.exec_module(entry)
        root = self.project / 'serial_runs/uam_dima_author_20261106_v1'
        root.mkdir(parents=True)
        vendor = self.project / 'workspace/vendor/DIMA_20261106'
        original = WORKSPACE / 'projects/uam-dima-author-20261106'
        for relative in entry.SOURCE_HASHES['dima']:
            target = vendor / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / relative, target)
        source_before = {r: repair.digest(vendor / r) for r in entry.SOURCE_HASHES['dima']}
        adapted = entry.wm_source(root, vendor)
        result = repair.patch_dima(self.project, entry, root, vendor)
        self.assertEqual(result['state'], 'APPLIED')
        self.assertEqual(source_before, {r: repair.digest(vendor / r) for r in source_before})
        # 原入口的完整性检查仍然认可新登记的适配哈希，并不关闭保护。
        self.assertEqual(entry.wm_source(root, vendor), adapted)
        again = repair.patch_dima(self.project, entry, root, vendor)
        self.assertEqual(result, again)
        backup = root / 'adapted_author/_engineering_backups' / repair.REVISION / 'world_model_env.py'
        self.assertEqual(repair.digest(backup), result['before_sha256'])
        self.assertEqual(repair.digest(adapted / repair.DIMA_RELATIVE), result['after_sha256'])

    def ppo_fixture(self, actual=100):
        root = self.project / 'serial_runs/uam_core_author_20261106_v1'
        item = dict(id='CORE__E1__K3__PPO_TF__s2102', arm='PPO_TF', steps=100)
        folder = root / 'cells' / item['id']
        folder.mkdir(parents=True)
        bank = root / 'bank/order_4301.csv'
        bank.parent.mkdir()
        bank.write_bytes(b'id,time\n1,0\n')
        job = dict(root=str(root), task=item, steps=100, marks=[50, 100], trace=str(bank),
                   bank=[dict(name='order_4301', path=str(bank), sha256=repair.digest(bank))])
        for step in (50, actual):
            stem = 'CP_' + str(step)
            model, norm = folder / (stem + '.zip'), folder / (stem + '.pkl')
            model.write_bytes(('immutable weights ' + str(step)).encode())
            with norm.open('wb') as stream:
                pickle.dump(FakeNorm(), stream)
            meta = dict(real_steps=step, decision_records=step * 2, model=model.name, norm=norm.name,
                        model_sha=repair.digest(model), norm_sha=repair.digest(norm))
            self.write(folder / (stem + '.json'), meta)
        self.write(folder / 'RESUME.json', meta)
        self.write(folder / 'TRAIN_END.json', dict(status='COMPLETE', real_steps=actual, seconds=1234))
        self.write(folder / 'LIVE.json', dict(real_steps=90))
        self.write(folder / 'END.json', dict(status='ENGINEERING_FAILED', real_steps=90, traceback=repair.SOURCE_ERROR))
        self.write(folder / 'JOB.json', job)
        return root, folder, item, job

    def test_full_training_not_lost_due_to_stale_live(self):
        _, folder, _, job = self.ppo_fixture()
        resume, selected = repair.select_checkpoints(folder, job)
        self.assertEqual(resume['real_steps'], 100)
        self.assertEqual([row['mark'] for row in selected], [50, 100])
        self.assertEqual(selected[-1]['pointer'], 'RESUME.json')
        self.assertEqual(repair.read(folder / 'LIVE.json')['real_steps'], 90)

    def test_missing_or_modified_checkpoint_does_not_retrain(self):
        _, folder, _, job = self.ppo_fixture()
        (folder / 'CP_50.zip').write_bytes(b'tampered')
        with self.assertRaises(RuntimeError):
            repair.select_checkpoints(folder, job)

    def test_unfinished_checkpoint_is_rejected(self):
        _, folder, _, job = self.ppo_fixture(actual=90)
        with self.assertRaises(RuntimeError):
            repair.select_checkpoints(folder, job)

    def test_ppo_worker_is_fresh_eval_only_and_never_changes_weights(self):
        root, folder, item, job = self.ppo_fixture()
        calls = dict(runtime=0, evaluate=0, predict=0, probe=0)
        before = {p.name: repair.digest(p) for p in folder.iterdir() if p.suffix in ('.zip', '.pkl')}
        class Phase:
            pack = 'pack'
            def __init__(self, *args): pass
            def close(self): pass
        def runtime(*args):
            calls['runtime'] += 1
            if calls['runtime'] != 1:
                raise RuntimeError('模拟二次环境加载错误')
            return object(), 'native'
        def load_model(path, device):
            self.assertEqual(device, 'cpu')
            class Predictor:
                def predict(self, obs, deterministic):
                    self.deterministic = deterministic
                    assert deterministic is False
                    calls['predict'] += 1
                    return [0], None
            return Predictor()
        def evaluate(base, m, r, t, bank, factory, mark, final):
            calls['evaluate'] += 1
            factory(123)([1], None)
            return [dict(trace='order_4301', checkpoint=mark, metrics=dict(fixed_cutoff_cost=42))]
        def probe(*args): calls['probe'] += 1
        def no_training(*args): raise AssertionError('不能调用训练')
        base = types.SimpleNamespace(runtime=runtime, gym_class=lambda m: Phase,
                                     torch_features=lambda pack: None, train_ppo=no_training)
        module = types.SimpleNamespace(lock_file=lambda p: types.SimpleNamespace(close=lambda: None),
            base_module=lambda: base, evaluate=evaluate, decision_probe=probe, save=repair.write_json)
        fake_modules = {
            'numpy': types.SimpleNamespace(asarray=lambda value: FakeArray()),
            'torch': types.SimpleNamespace(set_num_threads=lambda count: None),
            'stable_baselines3': types.SimpleNamespace(PPO=types.SimpleNamespace(load=load_model)),
        }
        with patch.object(repair, 'load_entry', return_value=(module, root, {})), \
             patch.object(repair, 'validate_job', return_value=(folder, job)), \
             patch.object(repair, 'own_workers', return_value=[]), patch.dict(sys.modules, fake_modules):
            repair.ppo_worker(self.project, item['id'])
        self.assertEqual(calls, dict(runtime=1, evaluate=2, predict=2, probe=1))
        self.assertEqual(before, {p.name: repair.digest(p) for p in folder.iterdir() if p.suffix in ('.zip', '.pkl')})
        self.assertEqual(repair.read(folder / 'END.json')['status'], 'COMPLETE')
        self.assertEqual(repair.read(folder / 'EVALUATION.json')['new_training_transitions'], 0)
        backup = root / 'engineering_repair' / repair.REVISION / 'ppo' / item['id'] / 'before/END.json'
        self.assertEqual(repair.read(backup)['status'], 'ENGINEERING_FAILED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
