"""阶段信用实验的纯合同/代数检查；不加载Torch，不启动仿真或训练。"""
import ast
import base64
import hashlib
import importlib.util
import pathlib
import subprocess
import sys
import unittest
import zlib


PATH = pathlib.Path(__file__).resolve().parents[1] / 'train_uam_stage_credit_18h_ONEFILE.py'
SPEC = importlib.util.spec_from_file_location('stage_contract_only', PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class StageContract(unittest.TestCase):
    def test_no_learning_library_imported(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('stable_baselines3', sys.modules)

    def test_sealed_sources_and_syntax(self):
        for blob, wanted in ((M.FROZEN_RUNTIME_B85, M.FROZEN_RUNTIME_SHA), (M.FROZEN_BASE_B85, M.FROZEN_BASE_SHA),
                             (M.FROZEN_STIN_B85, M.FROZEN_STIN_SHA)):
            raw = zlib.decompress(base64.b85decode(blob))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), wanted)
            ast.parse(raw.decode('utf-8-sig'))
        ast.parse(PATH.read_text(encoding='utf-8'))

    def test_budget_and_seed_blocks(self):
        tasks = M.tasks()
        self.assertEqual(M.STEPS // 20480, 119)
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len(tasks) * M.STEPS, 58490880)
        self.assertEqual(M.CAP, 60000000)
        self.assertLess(M.PLANNED_STEPS, M.CAP)
        for seed in (1901, 1902, 1903, 1904):
            block = [t for t in tasks if t['seed'] == seed]
            self.assertEqual(len(block), 6)
            self.assertEqual({t['arm'] for t in block if t['control'] == 'J3'}, set(M.ARMS) | {'STIN_ON','STIN_OFF'})
            self.assertEqual({t['stin_mode'] for t in block}, {'PLAIN','ON','OFF'})
            self.assertTrue(all(t['network'] == 'TRANSFORMER' and t['gamma'] == .99 for t in block))
        self.assertTrue(all(t['control'] == 'J3' for t in tasks[:20]))

    def test_stin_compatibility_patch_is_only_one_assignment(self):
        raw = zlib.decompress(base64.b85decode(M.FROZEN_STIN_B85))
        original = ast.parse(raw.decode('utf-8-sig'))
        adapted = M.stin_compatibility_tree()
        self.assertEqual(sum(isinstance(n, ast.Assign) for n in ast.walk(original)) -
                         sum(isinstance(n, ast.Assign) for n in ast.walk(adapted)), 1)
        self.assertEqual(sum(isinstance(n, ast.Call) for n in ast.walk(original)),
                         sum(isinstance(n, ast.Call) for n in ast.walk(adapted)))
        ast.fix_missing_locations(adapted)
        compile(adapted, '<合同检查，不执行STIN>', 'exec')

    def test_dynamic_module_can_enter_fresh_child(self):
        # 仅传递无学习库的小型替身，验证并行环境不会依赖不存在的模块文件。
        import cloudpickle
        raw = b'class Factory:\n def __call__(self): return 17\n'
        module = M.unpack('stage_pickle_contract_stub', base64.b85encode(zlib.compress(raw)), hashlib.sha256(raw).hexdigest())
        payload = cloudpickle.dumps(module.Factory())
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys,cloudpickle; x=cloudpickle.loads(sys.stdin.buffer.read()); print(x())'],
            input=payload, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', errors='replace'))
        self.assertEqual(result.stdout.strip(), b'17')

    def test_conditional_equals_split_without_baseline_difference(self):
        for p in (.3, .9, 1., 1.4, 2.):
            for r in (.5, 1., 1.7):
                for a in (-3., 0., 4.):
                    self.assertEqual(M.arm_loss_reference('SPLIT', p, r, a, 7, 7),
                                     M.arm_loss_reference('CONDITIONAL', p, r, a, 7, 7))

    def test_inactive_air_cannot_change_split_objective(self):
        for p in (.5, 1., 1.5):
            x = M.arm_loss_reference('CONDITIONAL', p, .1, 2., -10., 100., active_r=False)
            y = M.arm_loss_reference('CONDITIONAL', p, 4., 2., -10., -100., active_r=False)
            self.assertEqual(x, y)

    def test_original_and_split_have_same_first_order_gradient(self):
        # 未触发裁剪时，两种写法的常数不同，但在旧策略处每个头的一阶梯度相同。
        step = 1e-6
        for advantage in (-2., 3.):
            for head in (0, 1):
                def gradient(arm):
                    ratios = [1., 1.]; ratios[head] += step
                    high = M.arm_loss_reference(arm, *ratios, advantage, 0., 0.)
                    ratios[head] -= 2 * step
                    low = M.arm_loss_reference(arm, *ratios, advantage, 0., 0.)
                    return (high - low) / (2 * step)
                self.assertAlmostEqual(gradient('JOINT'), gradient('SPLIT'), places=7)

    def test_clipping_control_is_not_redundant(self):
        # 联合比率可因两个头反向变化而抵消；分头裁剪会改变有限步更新，因此必须保留中间对照。
        joint = M.arm_loss_reference('JOINT', 2., .5, 2., 0., 0.)
        split = M.arm_loss_reference('SPLIT', 2., .5, 2., 0., 0.)
        self.assertNotEqual(joint, split)

    def test_new_confirmation_orders_do_not_reuse_validation(self):
        source = PATH.read_text(encoding='utf-8')
        self.assertIn('(6501, 6502, 6503, 6504)', source)
        self.assertIn("event=event", source)
        self.assertIn("'air_only'", source)
        self.assertIn("cloudpickle.register_pickle_by_value(m)", source)


if __name__ == '__main__':
    unittest.main()
