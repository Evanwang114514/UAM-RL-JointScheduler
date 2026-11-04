"""STIN补齐脚本的纯合同检查；不导入Torch，不启动环境、推理或训练。"""

import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib


SCRIPT = Path(__file__).resolve().parents[1] / "train_uam_stin_E01_completion_ONEFILE.py"
SPEC = importlib.util.spec_from_file_location("stin_completion_contract", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class StinCompletionContract(unittest.TestCase):
    def test_import_does_not_load_learning_libraries(self):
        for name in ("torch", "jax", "stable_baselines3"):
            self.assertNotIn(name, sys.modules)

    def test_only_missing_single_cells_are_registered(self):
        rows = M.tasks()
        self.assertEqual(len(rows), 16)
        self.assertEqual({t["control"] for t in rows}, {"LQ"})
        self.assertEqual({t["k"] for t in rows}, {3})
        for environment in ("E0", "E1"):
            for seed in M.SEEDS:
                pair = [r for r in rows if r["environment"] == environment and r["seed"] == seed]
                self.assertEqual({r["stin_mode"] for r in pair}, {"ON", "OFF"})
                self.assertEqual(len(pair), 2)
                unchanged = [{k: v for k, v in r.items() if k not in ("id", "arm", "stin_mode", "run_order")} for r in pair]
                self.assertEqual(unchanged[0], unchanged[1])
        self.assertTrue(all(t["environment"] == "E0" for t in rows[:8]))
        self.assertEqual(len(M.legacy_ids()), 8)

    def test_budget_horizon_and_information_match_old_stin(self):
        self.assertEqual(M.STEPS, 2_437_120)
        self.assertEqual(M.NEW_STEPS, 16 * M.STEPS)
        self.assertEqual(M.NEW_STEPS, 38_993_920)
        self.assertLess(M.NEW_STEPS, M.CAP)
        self.assertEqual(M.STEPS % 20480, 0)
        for row in M.tasks():
            self.assertEqual((row["reward_mode"], row["time_mode"], row["gamma"]), ("PT_POST", "RAW", .99))
            self.assertTrue(row["timing"] and row["batch"])
            self.assertEqual(row["credit_arm"], "JOINT")
            self.assertEqual(row["fleet"], 12 if row["environment"] == "E1" else None)

    def test_sealed_runtime_is_exact_old_file(self):
        raw = zlib.decompress(base64.b85decode(M.SEALED_STAGE_B85))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), M.SEALED_STAGE_SHA)
        old = SCRIPT.parent / "train_uam_stage_credit_18h_ONEFILE.py"
        self.assertEqual(raw, old.read_bytes())
        ast.parse(raw.decode("utf-8-sig"))
        ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def test_sealed_stage_load_is_still_metadata_only(self):
        # 本项只核对标准库模块封装；不声称在本地验证了真实并行反序列化。
        stub = types.ModuleType("cloudpickle")
        stub.register_pickle_by_value = lambda module: None
        with patch.dict(sys.modules, {"cloudpickle": stub}):
            stage = M.load_stage()
        self.assertEqual(stage.ROOT_NAME, M.ROOT_NAME)
        self.assertEqual(stage.PLANNED_STEPS, M.NEW_STEPS)
        self.assertEqual(stage.CAP, M.CAP)
        self.assertEqual(stage.PROJECT, SCRIPT.parent)
        stage.stin_compatibility_tree()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("stable_baselines3", sys.modules)

    def fixture(self, root):
        # 构造已完成的旧记录；故意不构造真实模型，检查“记录全”和“模型全”不混淆。
        suite = dict(script_sha256=M.LEGACY_ENTRY_SHA, frozen_stin_sha256=M.STIN_SHA)
        (root / "SUITE_PLAN.json").write_text(json.dumps(suite), encoding="utf-8")
        for ident in M.legacy_ids():
            folder = root / "cells" / ident
            folder.mkdir(parents=True)
            mode = "OFF" if "__STIN_OFF__" in ident else "ON"
            task = dict(id=ident, environment="E1", control="J3", k=3, fleet=12,
                        seed=int(ident.rsplit("s", 1)[-1]), stin_mode=mode,
                        reward_mode="PT_POST", gamma=.99, timing=True, batch=True)
            entries = {
                "TRAIN_END.json": dict(steps=M.STEPS),
                "STAGE_EVAL_END.json": dict(status="COMPLETE", step=M.STEPS, weights_unchanged=True),
                "MODEL_MANIFEST.json": dict(task=task, stin_source_sha256=M.STIN_SHA, config=dict(actual_steps=M.STEPS)),
                "RESULT.json": dict(main_late_confirmation=dict(rpt=1.)),
                "NEW_ORDER_CONFIRMATION.json": dict(episodes=[{} for _ in range(4)]),
            }
            for name, value in entries.items():
                (folder / name).write_text(json.dumps(value), encoding="utf-8")

    def test_old_completed_records_are_not_marked_missing_training(self):
        with tempfile.TemporaryDirectory(prefix="uam_stin_test_", dir=SCRIPT.parent / "tests") as temporary:
            root = Path(temporary) / M.LEGACY_NAME
            root.mkdir()
            self.fixture(root)
            before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.json")}
            result = M.audit_legacy(root, Path(temporary) / "missing.tar.gz")
            self.assertEqual(result["complete"], 8)
            self.assertEqual(result["final_models_present"], 0)
            after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.json")}
            self.assertEqual(before, after)

    def test_wrong_old_protocol_is_not_reused(self):
        with tempfile.TemporaryDirectory(prefix="uam_stin_test_", dir=SCRIPT.parent / "tests") as temporary:
            root = Path(temporary) / M.LEGACY_NAME
            root.mkdir()
            self.fixture(root)
            ident = M.legacy_ids()[0]
            path = root / "cells" / ident / "MODEL_MANIFEST.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["task"]["reward_mode"] = "SOURCE"
            path.write_text(json.dumps(record), encoding="utf-8")
            result = M.audit_legacy(root, Path(temporary) / "missing.tar.gz")
            self.assertEqual(result["complete"], 7)

    def test_archive_fallback_never_extracts_files(self):
        with tempfile.TemporaryDirectory(prefix="uam_stin_test_", dir=SCRIPT.parent / "tests") as temporary:
            base = Path(temporary)
            root = base / M.LEGACY_NAME
            root.mkdir()
            self.fixture(root)
            archive = base / "old_data.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(root, arcname=M.LEGACY_NAME)
            absent = base / "read_only_absent"
            result = M.audit_legacy(absent, archive)
            self.assertEqual(result["complete"], 8)
            self.assertFalse(absent.exists())

    def test_existing_train_end_without_models_is_not_silently_retrained(self):
        with tempfile.TemporaryDirectory(prefix="uam_stin_test_", dir=SCRIPT.parent / "tests") as temporary:
            folder = Path(temporary)
            task = M.tasks()[0]
            (folder / "TRAIN_END.json").write_text(json.dumps(dict(task=task, steps=M.STEPS)), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                M.train_complete(folder, task)

    def test_one_corrupt_cell_does_not_break_suite_aggregation(self):
        with tempfile.TemporaryDirectory(prefix="uam_stin_test_", dir=SCRIPT.parent / "tests") as temporary:
            root = Path(temporary)
            task = M.tasks()[0]
            folder = root / "cells" / task["id"]
            folder.mkdir(parents=True)
            (folder / "TRAIN_END.json").write_text(json.dumps(dict(task=task, steps=17)), encoding="utf-8")
            fake = types.SimpleNamespace(budget_account=lambda unused: (17, 20480), save_csv=lambda *args: None)
            progress = M.aggregate(fake, root, M.tasks(), dict(complete=8, final_models_present=8))
            self.assertEqual(progress["failed"], 1)
            self.assertEqual(progress["trained"], 0)
            self.assertEqual(progress["total"], 16)


if __name__ == "__main__":
    unittest.main()
