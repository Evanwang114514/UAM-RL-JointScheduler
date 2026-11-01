"""仅测试清单、路径和日志核验；从不执行真实删除入口。"""
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
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


class ProcessRaceTests(unittest.TestCase):
    def process(self, *, error=None, data=b"python\0train_uam_round3_uagmc_causal_ONEFILE.py\0"):
        process = mock.MagicMock()
        process.name = "12345"
        process.stat.return_value = SimpleNamespace(st_uid=998, st_ino=42)
        handle = process.__truediv__.return_value
        if error:
            handle.read_bytes.side_effect = error
        else:
            handle.read_bytes.return_value = data
        return process

    def test_exited_before_stat_is_not_unreadable(self):
        process = self.process()
        process.stat.side_effect = FileNotFoundError("进程已退出")
        self.assertEqual(module.inspect_process(process, 998)[0], "gone")

    def test_exited_between_stat_and_cmdline_is_not_unreadable(self):
        process = self.process(error=FileNotFoundError("cmdline已消失"))
        process.stat.side_effect = [SimpleNamespace(st_uid=998, st_ino=42), FileNotFoundError("进程已退出")]
        self.assertEqual(module.inspect_process(process, 998)[0], "gone")

    def test_live_permission_denied_still_blocks(self):
        process = self.process(error=PermissionError("权限不足"))
        status, details = module.inspect_process(process, 998)
        self.assertEqual(status, "unreadable")
        self.assertIn("PermissionError", details[0])

    def test_missing_cmdline_with_live_pid_still_blocks(self):
        process = self.process(error=FileNotFoundError("cmdline缺失但PID仍在"))
        self.assertEqual(module.inspect_process(process, 998)[0], "unreadable")

    def test_actual_reference_still_blocks_with_pid(self):
        process = self.process()
        root = mock.Mock()
        root.iterdir.return_value = [process]
        with mock.patch.object(module.os, "getuid", return_value=998, create=True):
            active, unreadable = module.precise_process_references("uam_round3_uagmc_causal_V1",
                {"train_uam_round3_uagmc_causal_ONEFILE.py"}, root)
        self.assertEqual(active, [12345])
        self.assertEqual(unreadable, [])

    def test_process_identity_change_is_not_ignored(self):
        process = self.process()
        process.stat.side_effect = [SimpleNamespace(st_uid=998, st_ino=value) for value in (42, 43, 44, 45)]
        self.assertEqual(module.inspect_process(process, 998)[0], "unreadable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
