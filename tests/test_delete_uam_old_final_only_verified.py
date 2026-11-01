"""仅测试清单、路径和日志核验；从不执行真实删除入口。"""
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("delete_audit", ROOT / "delete_uam_old_final_only_verified.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.name = "uam_100m_two_tracks_v1"
        cell = "A__OD_LQ__E4__K2__s711"
        folder = PurePosixPath("/home/SY/uam/UAGMC-main/serial_runs") / self.name / cell
        self.rows = [dict(experiment=self.name, cell=cell, action="CANDIDATE", bytes="100", uid="998", nlink="1",
                          path=str(folder / "checkpoints" / filename))
                     for filename in ("uam_ppo_100_steps.zip", "uam_ppo_vecnormalize_100_steps.pkl")]
        self.all_rows = copy.deepcopy(self.rows) + [dict(action="KEEP_FINAL", path=str(folder / "final_model.zip"))]
        self.patches = [mock.patch.object(module, "EXPECTED_FILES", 2), mock.patch.object(module, "EXPECTED_BYTES", 200),
                        mock.patch.object(module, "EXPECTED_KEEP", 1), mock.patch.object(module, "GROUPS", {self.name: 2})]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def test_exact_manifest_passes_without_deletion(self):
        module.validate_manifest(self.rows, self.all_rows)

    def test_unknown_experiment_stops(self):
        self.rows[0]["experiment"] = "others_project"
        with self.assertRaises(RuntimeError):
            module.validate_manifest(self.rows, self.all_rows)

    def test_final_path_cannot_be_target(self):
        self.rows[0]["path"] = self.all_rows[-1]["path"]
        self.all_rows[0]["path"] = self.rows[0]["path"]
        with self.assertRaises(RuntimeError):
            module.validate_manifest(self.rows, self.all_rows)

    def test_path_traversal_stops(self):
        self.rows[0]["cell"] = "../../outside"
        with self.assertRaises(RuntimeError):
            module.validate_manifest(self.rows, self.all_rows)

    def test_hardlink_or_different_uid_stops(self):
        for field, value in (("nlink", "2"), ("uid", "1001")):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaises(RuntimeError):
                module.validate_manifest(rows, self.all_rows)

    def test_duplicate_target_stops(self):
        rows = [self.rows[0], self.rows[0]]
        with self.assertRaises(RuntimeError):
            module.validate_manifest(rows, self.all_rows)

    def test_size_change_stops(self):
        self.rows[0]["bytes"] = "101"
        with self.assertRaises(RuntimeError):
            module.validate_manifest(self.rows, self.all_rows)

    def test_partial_pair_can_resume_only_with_verified_final(self):
        missing = [self.rows[0]]
        remaining = [self.rows[1]]
        final = dict(self.all_rows[-1], experiment=self.name, cell=self.rows[0]["cell"])
        module.check_current_candidates([final], remaining, missing)
        with self.assertRaises(RuntimeError):
            module.check_current_candidates([], remaining, missing)

    def test_unexplained_orphan_or_extra_target_stops(self):
        final = dict(self.all_rows[-1], experiment=self.name, cell=self.rows[0]["cell"])
        with self.assertRaises(RuntimeError):
            module.check_current_candidates([final], [self.rows[1]], [])
        extra = dict(self.rows[0], path=self.rows[0]["path"].replace("100_steps", "999_steps"))
        with self.assertRaises(RuntimeError):
            module.check_current_candidates([final, extra], self.rows, [])


class PackageAndJournalTests(unittest.TestCase):
    def test_actual_downloaded_package(self):
        archive = Path(r"E:\Study Files\github\UAM-predict\UAGMC-main\experiment_archives\old_final_only_20261101_211457_1905767_AUDIT_ONLY.tar.gz")
        if not archive.is_file():
            self.skipTest("当前机器没有用户下载的原审计包")
        candidates, all_rows, proofs, _, _ = module.load_package(archive)
        module.validate_manifest(candidates, all_rows)
        self.assertEqual(len(candidates), 27386)
        self.assertEqual(len(proofs), 561)

    def test_journal_cannot_refer_to_different_manifest(self):
        with tempfile.TemporaryDirectory(prefix="delete_manifest_test_", dir=ROOT) as temp:
            folder = Path(temp).resolve()
            self.assertEqual(folder.parent, ROOT.resolve())
            path = folder / "journal.jsonl"
            path.write_text(json.dumps(dict(package_sha256="different", action="intent", row={})) + "\n", encoding="utf-8")
            with mock.patch.object(module.os, "getuid", return_value=path.stat().st_uid, create=True):
                with self.assertRaises(RuntimeError):
                    module.prior_intents(path, {})

    def test_exact_journal_can_resume(self):
        with tempfile.TemporaryDirectory(prefix="delete_manifest_test_", dir=ROOT) as temp:
            folder = Path(temp).resolve()
            self.assertEqual(folder.parent, ROOT.resolve())
            path = folder / "journal.jsonl"
            row = dict(path="/fixed/checkpoint.zip", inode="123")
            path.write_text(json.dumps(dict(package_sha256=module.PACKAGE_SHA, action="intent", row=row)) + "\n", encoding="utf-8")
            with mock.patch.object(module.os, "getuid", return_value=path.stat().st_uid, create=True):
                self.assertEqual(module.prior_intents(path, {row["path"]: row}), {row["path"]})
                with self.assertRaises(RuntimeError):
                    module.prior_intents(path, {row["path"]: dict(row, inode="456")})


if __name__ == "__main__":
    unittest.main(verbosity=2)
