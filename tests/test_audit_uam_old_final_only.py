"""仅用临时小文件验证审计分类，不运行任何实验或神经网络。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

SOURCE = Path(__file__).resolve().parents[1] / "audit_uam_old_final_only.py"
spec = importlib.util.spec_from_file_location("old_audit", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uam_readonly_audit_test_", dir=SOURCE.parent)
        self.project = Path(self.temp.name).resolve()
        self.assertEqual(self.project.parent, SOURCE.parent.resolve())
        self.name = "uam_21x7_700k_METHOD_FIRST_V2"
        self.cell = self.project / "serial_runs" / self.name / "TEST__s601"
        self.cell.mkdir(parents=True)
        self.cp = self.cell / "checkpoints"
        self.cp.mkdir()
        self.record = {"method": "UAGMC", "seed": 601, "steps": 200}
        self.raw = json.dumps(self.record).encode()
        (self.cell / "manifest.json").write_bytes(self.raw)
        _, raw_sha, canonical_sha = module.read_record(self.cell / "manifest.json")
        self.evidence = {"anchors": [dict(raw=raw_sha, canonical=canonical_sha,
                                          archive="known.zip", record="known/manifest.json")], "scripts": []}
        self.uid = mock.patch.object(module.os, "getuid", return_value=self.cell.stat().st_uid, create=True)
        self.uid.start()
        self.process = mock.patch.object(module, "process_references", return_value=([], []))
        self.process_mock = self.process.start()

    def tearDown(self):
        self.process.stop()
        self.uid.stop()
        self.temp.cleanup()

    def model(self, path, step):
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("data", json.dumps({"num_timesteps": step}))
            bundle.writestr("policy.pth", b"fixture-only-not-a-real-network")

    def standard(self):
        for step in (100, 200):
            self.model(self.cp / f"uam_ppo_{step}_steps.zip", step)
            (self.cp / f"uam_ppo_vecnormalize_{step}_steps.pkl").write_bytes(b"fixture")
        self.model(self.cell / "final_model.zip", 200)
        (self.cell / "final_vecnormalize.pkl").write_bytes(b"fixture")

    def audit(self):
        return module.audit_run(self.project, self.name, self.evidence)

    def test_owned_final_only(self):
        self.standard()
        before = {str(p): p.read_bytes() for p in self.cell.rglob("*") if p.is_file()}
        result, rows, matches, _ = self.audit()
        self.assertEqual(result["candidate_files"], 4)
        self.assertEqual(result["retained_final_files"], 2)
        self.assertEqual(len(matches), 1)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.cell.rglob("*") if p.is_file()})

    def test_unverified_metadata_is_retained(self):
        self.standard()
        (self.cell / "manifest.json").write_text(json.dumps(dict(self.record, seed=999)), encoding="utf-8")
        self.assertEqual(self.audit()[0]["candidate_files"], 0)

    def test_no_final_is_retained(self):
        self.standard()
        # 删除的只是本测试自己的临时文件，不涉及真实实验目录。
        (self.cell / "final_model.zip").unlink()
        self.assertEqual(self.audit()[0]["candidate_files"], 0)

    def test_missing_pair_is_not_candidate(self):
        self.standard()
        (self.cp / "uam_ppo_vecnormalize_100_steps.pkl").unlink()
        result, rows, _, _ = self.audit()
        self.assertEqual(result["candidate_files"], 2)
        row = next(item for item in rows if item["path"].endswith("uam_ppo_100_steps.zip"))
        self.assertEqual(row["action"], "REVIEW")

    def test_root_plan_verifies_registered_cells_only(self):
        self.standard()
        plan = self.cell.parent / "PLAN.json"
        plan.write_text(json.dumps({"schema": "known", "cells": [self.cell.name]}), encoding="utf-8")
        _, raw, canonical = module.read_record(plan)
        proof = {"anchors": [dict(raw=raw, canonical=canonical, archive="plan.zip", record="PLAN.json",
                                   registered_cells={self.cell.name: 200})]}
        self.assertEqual(module.audit_run(self.project, self.name, proof)[0]["candidate_files"], 4)
        proof["anchors"][0]["registered_cells"] = {"other": 200}
        self.assertEqual(module.audit_run(self.project, self.name, proof)[0]["candidate_files"], 0)

    def test_last_checkpoint_with_completion_retained(self):
        for step in (100, 200):
            self.model(self.cp / f"checkpoint_{step}.model.zip", step)
            (self.cp / f"checkpoint_{step}.norm.pkl").write_bytes(b"fixture")
            (self.cp / f"checkpoint_{step}.state.pt").write_bytes(b"fixture")
            (self.cp / f"checkpoint_{step}.json").write_text(json.dumps(dict(step=step, ready=True,
                model=f"checkpoint_{step}.model.zip", norm=f"checkpoint_{step}.norm.pkl",
                state=f"checkpoint_{step}.state.pt")), encoding="utf-8")
        (self.cell / "TRAIN_END.json").write_text(json.dumps(dict(status="COMPLETE", actual_steps=200)), encoding="utf-8")
        result = self.audit()[0]
        self.assertEqual(result["candidate_files"], 3)
        self.assertEqual(result["retained_final_files"], 3)

    def test_final_old_or_wrong_target_is_retained(self):
        self.standard()
        self.model(self.cell / "final_model.zip", 100)
        self.assertEqual(self.audit()[0]["candidate_files"], 0)

    def test_active_run_is_retained(self):
        self.standard()
        self.process_mock.return_value = ([1234], [])
        self.assertEqual(self.audit()[0]["candidate_files"], 0)

    def test_unreadable_process_report_does_not_claim_execution_clearance(self):
        self.standard()
        self.process_mock.return_value = ([], [1234])
        result = self.audit()[0]
        self.assertEqual(result["candidate_files"], 4)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["unreadable_processes"], [1234])

    def source_proof(self):
        script = self.project / "owned_source.py"
        script.write_bytes(b"fixture source only")
        fields = dict(version="TEST", code_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                      engine_sha256="known-engine", freeze_sha256="known-freeze")
        registered = {self.cell.name: dict(cell=["E3", 2, "LQ", "UAGMC"], seed=601)}
        registered.update({f"other_{i}": dict(cell=["fixture", i, "LQ", "UAGMC"], seed=601) for i in range(83)})
        (self.cell.parent / "PLAN.json").write_text(json.dumps(dict(fields, steps_per_run=200,
            specs=list(registered.values()))), encoding="utf-8")
        (self.cell / "contract.json").write_text(json.dumps(dict(fields, **registered[self.cell.name], steps=200)),
                                                  encoding="utf-8")
        return {"anchors": [], "source_contract": dict(script=dict(name=script.name, sha256=fields["code_sha256"]),
                fields=fields, steps=200, registered_cells=registered)}

    def test_source_plan_and_contract_all_match(self):
        self.standard()
        proof = self.source_proof()
        result, _, matches, _ = module.audit_run(self.project, self.name, proof)
        self.assertEqual(result["candidate_files"], 4)
        self.assertEqual(matches[0]["scope"], "source_contract")

    def test_source_hash_mismatch_is_retained(self):
        self.standard()
        proof = self.source_proof()
        (self.project / "owned_source.py").write_bytes(b"different file")
        self.assertEqual(module.audit_run(self.project, self.name, proof)[0]["candidate_files"], 0)

    def test_source_contract_wrong_seed_is_retained(self):
        self.standard()
        proof = self.source_proof()
        contract = self.cell / "contract.json"
        document = json.loads(contract.read_bytes())
        document["seed"] = 999
        contract.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(module.audit_run(self.project, self.name, proof)[0]["candidate_files"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
