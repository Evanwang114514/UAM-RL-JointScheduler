# -*- coding: utf-8 -*-
"""标准库检查；不加载真实深度学习库、不训练、不分配GPU。"""
import ast
import importlib.util
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
AUTHOR = WORKSPACE / 'projects/uam-marie-author-20261106'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repair = load(PROJECT / 'repair_uam_marie_nan_20261107_ONEFILE.py', 'uam_marie_repair_test')
original = load(PROJECT / repair.ENTRY_NAME, 'uam_marie_original_test')


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='uam_marie_repair_', dir=WORKSPACE / 'scratch')
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.root = self.project / 'serial_runs' / repair.RUN_NAME
        self.audit = self.root / 'engineering_repair' / repair.REVISION

    def patches(self):
        return repair.patch_text(*((AUTHOR / name).read_text(encoding='utf-8-sig')
                                   for name in (repair.CRITIC_FILE, repair.LEARNER_FILE)))

    def test_original_entry_and_budget_contract(self):
        entry = self.project / repair.ENTRY_NAME
        shutil.copy2(PROJECT / repair.ENTRY_NAME, entry)
        self.assertEqual(repair.sha(entry), repair.ENTRY_SHA)
        plan = original.plan_make()
        plan['entrypoint_sha256'] = repair.ENTRY_SHA
        repair.write_json(self.root / 'PLAN.json', plan)
        module, _, _ = repair.load_entry(self.project)
        self.assertEqual(module.plan_make(), original.plan_make())
        self.assertEqual(plan['planned_real_steps'], 24_084_480)
        self.assertEqual(plan['budget_cap'], 25_000_000)

    def test_math_only_wraps_two_original_critic_calls(self):
        patches = self.patches()
        old = ast.parse((AUTHOR / repair.CRITIC_FILE).read_text(encoding='utf-8-sig'))
        new = ast.parse(patches[repair.CRITIC_FILE])
        def is_math_wrapper(node):
            return any(ast.unparse(item.context_expr) == 'sdpa_kernel(SDPBackend.MATH)' for item in node.items)

        wrappers = [n for n in ast.walk(new) if isinstance(n, ast.With) and is_math_wrapper(n)]
        self.assertEqual(len(wrappers), 2)

        class StripPatch(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == 'torch.nn.attention' else node

            def visit_With(self, node):
                return node.body if is_math_wrapper(node) else self.generic_visit(node)

        self.assertEqual(ast.dump(old), ast.dump(StripPatch().visit(new)))

    def test_real_source_patch_idempotent_and_original_checks_pass(self):
        vendor = self.project / 'workspace/vendor/MARIE_20261106'
        before = original.SOURCE_HASHES['marie']
        for relative in before:
            out = vendor / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(AUTHOR / relative, out)
        adapted = original.wm_source(self.root, vendor)
        # 模拟补丁仅写了一半时重新启动，应从保留的原清单安全完成。
        path = adapted / 'UAM_ADAPTATION.json'
        backup = self.root / 'adapted_author' / '_engineering_backups' / repair.REVISION
        repair.preserve(path, backup / 'UAM_ADAPTATION.before.json')
        repair.write_bytes(adapted / repair.CRITIC_FILE, self.patches()[repair.CRITIC_FILE])
        repair.install_patch(self.project, original, self.root, vendor, self.audit)
        repaired = repair.read(path)
        self.assertEqual(original.wm_source(self.root, vendor), adapted)
        repair.install_patch(self.project, original, self.root, vendor, self.audit)
        self.assertEqual(repair.read(path), repaired)
        self.assertEqual(len(repaired['engineering_patches']), 1)
        self.assertEqual(before, {relative: repair.sha(vendor / relative) for relative in before})

    def checkpoint(self, real, data):
        cell = self.root / 'cells' / repair.FAILED_CELL
        file = cell / 'replay' / f'RESUME_FULL_{real:09d}.pkl'
        repair.write_bytes(file, data)
        repair.write_json(cell / 'RESUME.json', dict(real_steps=real, decision_records=2 * real,
            full_file=file.name, full_sha256=repair.sha(file)))
        repair.write_json(cell / 'END.json', dict(status='ENGINEERING_FAILED', real_steps=87150))
        return file

    def test_exact_checkpoint_preserved_and_later_resume_does_not_add_backup(self):
        source = self.checkpoint(86673, b'full-state-immutable')
        # Windows工作区限制硬链接；用替身验证只会调用一次保护操作。
        with patch.object(repair.os, 'link', side_effect=shutil.copyfile) as link:
            self.assertEqual(repair.protect_resume(self.project, self.root, self.audit), 86673)
            link.assert_called_once()
        saved = repair.read(self.audit / 'RECOVERY.json')
        backup = Path(saved['protected'])
        # 轮换只删除原路径；最初恢复点仍可读。第二次不额外保留大回放。
        source.unlink()
        self.checkpoint(87500, b'new-full-state')
        self.assertEqual(repair.protect_resume(self.project, self.root, self.audit), 87500)
        self.assertEqual(backup.read_bytes(), b'full-state-immutable')
        self.assertEqual(len(list(backup.parent.glob('*.pkl'))), 1)
        self.assertEqual(repair.read(self.audit / 'RESUME.before.json')['real_steps'], 86673)

    def test_corrupted_resume_is_rejected(self):
        source = self.checkpoint(86673, b'full-state-immutable')
        source.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, '哈希不符'):
            repair.protect_resume(self.project, self.root, self.audit)

    def test_fail_fast_does_not_step_optimizer_on_nonfinite_values(self):
        tree = ast.parse(self.patches()[repair.LEARNER_FILE])
        learner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DreamerLearner')
        method = next(n for n in learner.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_optimizer')
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_uam_marie_finite')
        calls = []

        class Loss:
            def __init__(self, finite): self.finite = finite
            def detach(self): return self
            def backward(self): calls.append('backward')

        def reject_grad(*args, **kwargs):
            self.assertTrue(kwargs['error_if_nonfinite'])
            raise RuntimeError('bad gradient')

        fake = types.SimpleNamespace(isfinite=lambda loss: types.SimpleNamespace(all=lambda: loss.finite),
            nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=reject_grad)))
        scope = {'torch': fake}
        exec(compile(ast.Module(body=[function, method], type_ignores=[]), '<finite-guard>', 'exec'), scope)
        opt = types.SimpleNamespace(zero_grad=lambda: calls.append('zero_grad'), step=lambda: calls.append('step'))
        model = types.SimpleNamespace(parameters=lambda: [])
        with self.assertRaisesRegex(RuntimeError, '非有限值'):
            scope['apply_optimizer'](None, opt, model, Loss(False), 100.)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, 'bad gradient'):
            scope['apply_optimizer'](None, opt, model, Loss(True), 100.)
        self.assertEqual(calls, ['zero_grad', 'backward'])

    def test_old_marie_is_paused_without_process_signals(self):
        self.root.mkdir(parents=True)
        with patch.object(repair, 'own_processes', side_effect=[[123], [], []]), \
             patch.object(repair.os, 'kill', side_effect=AssertionError('不发送信号')):
            lock, marker_state = repair.pause_and_lock(original, self.root, self.project / repair.ENTRY_NAME, 30)
        lock.close()
        self.assertIn(repair.REVISION.encode(), marker_state[1])
        self.assertTrue((self.root / 'STOP_REQUESTED').is_file())

    def test_escape_is_rejected(self):
        with self.assertRaises((RuntimeError, ValueError)):
            repair.safe_path(self.project, self.project / '../outside', False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
