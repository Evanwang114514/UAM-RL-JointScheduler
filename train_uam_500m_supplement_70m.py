# -*- coding: utf-8 -*-
"""针对 500M 结果的约 70M 步配对补训，不改变物理环境或奖励。

K0/K1：S、AC、S+AC 在相同训练种子下比较原 PPO 与 target_kl=0.05。
C0/C1：在新种子下比较 J4+S 与条件式 J4C+S。
训练先完整结束，再做稀疏检查点回放；单格失败会记录并继续。
这批实验从头训练，不冒充没有随分析包提供的 500M 权重续训。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tarfile
import time
import traceback
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

import train_uam_500m_continuous as previous
import train_uam_500m_stagewise as core


ROOT = Path(__file__).resolve().parent
ROLLOUT = 20_480
KL_LIMIT = 0.05
REFERENCE_TRACE_SHA256 = "4e8b6ff6201a22051210782a54f808cb2f88e711a64b65d39f0c34aafe06730b"
DEFAULT_RUN_DIR = ROOT / "serial_runs" / "uam500m_supplement70m_s3"
REFERENCE_ARCHIVE = ROOT / "analysis_bundles" / "uam500m_core_20261006.tar.gz"
REFERENCE_MEMBER = "UAGMC-main/train_data/passengers_300.csv"
SOURCE_FILES = (
    "train_uam_500m_continuous.py",
    "train_uam_500m_stagewise.py",
    "uam500m_methods.py",
    "train_uam_pre5b_80m_autofunnel.py",
    "train_uam_60m_literature_matrix_v3_1.py",
    "train_uam_60m_jointfirst_v4_deferred_joint_eval.py",
    "train_uam_60m_jointfirst_v5_minimal_reposition.py",
    "train_uam_nextgen_100m_matrix.py",
    "train_uam_7x12_600k_v2.py",
)


def make_plan() -> list[core.Cell]:
    """优先执行更新幅度因果对照，再复核条件式 J4；配对格相邻运行。"""
    plan: list[core.Cell] = []
    for method in ("S_TDM_EVENT_FUSION", "A_ICM_AC", "SA_TDMFUSION_ICM"):
        for seed in range(501, 506):
            for stage in ("K0", "K1"):
                plan.append(core.Cell(stage, method, "J4", seed, 90 * ROLLOUT))
    for seed in range(511, 515):
        plan.append(core.Cell("C0", "S_TDM_EVENT_FUSION", "J4", seed, 89 * ROLLOUT))
        plan.append(core.Cell("C1", "J4C_S", "J4", seed, 89 * ROLLOUT))
    assert len(plan) == 38
    assert len({cell.key for cell in plan}) == len(plan)
    assert sum(cell.steps for cell in plan) == 69_877_760
    return plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_csv_rows(raw: bytes) -> list[list[str]]:
    """只消除 BOM 和换行差异，不悄悄修改乘客字段或顺序。"""
    return list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))


def verify_trace(trace: Path) -> dict[str, Any]:
    """核对 500M 乘客表；字节不同但逐行相同时允许运行。"""
    if not trace.is_file():
        raise FileNotFoundError(f"训练乘客文件不存在：{trace}")
    actual_sha = sha256(trace)
    match = "EXACT_BYTES"
    if actual_sha != REFERENCE_TRACE_SHA256:
        if not REFERENCE_ARCHIVE.is_file():
            raise RuntimeError(
                "乘客文件哈希与 500M 不同，且缺少参考分析包，无法确认训练数据一致。"
                f"请放入 {REFERENCE_ARCHIVE}，或恢复 500M 原始 passengers_300.csv。"
            )
        with tarfile.open(REFERENCE_ARCHIVE, "r:gz") as archive:
            member = archive.extractfile(REFERENCE_MEMBER)
            if member is None:
                raise RuntimeError("500M 分析包缺少原始训练乘客表")
            reference_raw = member.read()
        if hashlib.sha256(reference_raw).hexdigest() != REFERENCE_TRACE_SHA256:
            raise RuntimeError("500M 分析包内乘客表哈希与历史 manifest 不符")
        if normalized_csv_rows(trace.read_bytes()) != normalized_csv_rows(reference_raw):
            raise RuntimeError("当前训练乘客记录与 500M 不同，拒绝混合物理需求")
        match = "SAME_CSV_ROWS_DIFFERENT_BYTES"
    return {"path": str(trace), "sha256": actual_sha,
            "500m_reference_sha256": REFERENCE_TRACE_SHA256, "match": match}


def stage_kl(stage: str) -> float | None:
    return KL_LIMIT if stage == "K1" else None


def verify_saved_kl(root: Path, cell: core.Cell) -> None:
    """防止续跑时从错误的 PPO 变体或错配检查点恢复。"""
    checkpoint = core.latest_checkpoint(core.cell_dir(root, cell))
    if checkpoint is None:
        return
    step, model_path, _ = checkpoint
    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data"))
    actual = data.get("target_kl")
    expected = stage_kl(cell.stage)
    if (actual is None) != (expected is None):
        raise RuntimeError(f"{cell.key} 的 {step} 步检查点 target_kl 不匹配：{actual} != {expected}")
    if expected is not None and not math.isclose(float(actual), expected, abs_tol=1e-12):
        raise RuntimeError(f"{cell.key} 的检查点 target_kl 不匹配：{actual} != {expected}")


def source_fingerprints() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def manifest_for(plan: list[core.Cell], profile: Any, config: dict[str, Any],
                 trace_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "purpose": "500M_POSTHOC_SUPPLEMENT_FRESH_MATCHED_RUNS",
        "reference_commit": "c280cc1c1d73d4b78345ea5ae24bf26dad793609",
        "baseline_comparison_warning": "历史 MPTC 85.33 与当前复跑尚未对齐，不能据此判断胜负",
        "evaluation_boundary": "训练 trace 上的 DET 回放仅作诊断，不是独立最终测试",
        "trace": trace_contract,
        "source_sha256": source_fingerprints(),
        "profile": asdict(profile),
        "configuration": config,
        "ppo_factor": {"control_target_kl": None, "guarded_target_kl": KL_LIMIT,
                       "all_other_ppo_settings": "same_as_500m"},
        "plan": [asdict(cell) for cell in plan],
        "planned_training_steps": sum(cell.steps for cell in plan),
        "training_replicate": "independent_optimization_seed",
        "paired_seed_groups": {"K": [501, 502, 503, 504, 505],
                               "C": [511, 512, 513, 514]},
    }


def ensure_manifest(root: Path, manifest: dict[str, Any], *, resume: bool) -> None:
    path = root / "supplement70m_manifest.json"
    existing = core.read_json(path, None)
    if existing is not None:
        if existing != manifest:
            raise RuntimeError("现有目录的计划、乘客数据或依赖源码发生变化，拒绝混训；请换 --run-dir")
        if not resume:
            raise RuntimeError("运行目录已存在；续跑请显式添加 --resume")
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("运行目录非空但没有本实验 manifest，拒绝覆盖")
        root.mkdir(parents=True, exist_ok=True)
        core.write_json(path, manifest)


def complete(root: Path, cell: core.Cell) -> bool:
    record = core.read_json(core.cell_dir(root, cell) / "run_end.json", {})
    return record.get("status") == "COMPLETE" and int(record.get("actual_steps", 0)) >= cell.steps


def log_failure(root: Path, cell: core.Cell, phase: str, exc: Exception) -> None:
    previous.append_event(root / "supplement70m_failures.jsonl", {
        "time": time.time(), "phase": phase, "cell": asdict(cell),
        "error": repr(exc), "traceback": traceback.format_exc(),
    })


def run_training(root: Path, plan: list[core.Cell], profile: Any, device: str) -> None:
    failed: list[str] = []
    consecutive_failures = 0
    for index, cell in enumerate(plan, 1):
        if complete(root, cell):
            print(f"[SKIP {index}/{len(plan)}] {cell.key} 已完整训练", flush=True)
            consecutive_failures = 0
            continue
        print(f"[TRAIN {index}/{len(plan)}] {cell.key} | {cell.steps:,} 步 | "
              f"target_kl={stage_kl(cell.stage)}", flush=True)
        try:
            verify_saved_kl(root, cell)
            core.run_cell(root, cell, {}, profile, device, retry_failed=True)
            if not complete(root, cell):
                raise RuntimeError("训练返回但未保存完整终点")
            consecutive_failures = 0
        except Exception as exc:
            failed.append(cell.key)
            consecutive_failures += 1
            log_failure(root, cell, "train", exc)
            print(f"[FAILED] {cell.key}: {exc!r}；继续下一格", flush=True)
            if consecutive_failures >= 3:
                raise RuntimeError(
                    "连续三格失败，疑似系统性故障；已保留检查点，修复后加 --resume 继续"
                ) from exc
    if failed:
        print(f"[TRAIN END] 有 {len(failed)} 格失败，其余已完成；失败格可用 --resume 重试", flush=True)


def valid_evaluation(root: Path, cell: core.Cell) -> dict[str, Any] | None:
    analysis = core.cell_dir(root, cell) / "analysis"
    summary = core.read_json(analysis / "sparse_summary.json", {})
    curve = core.read_json(analysis / "sparse_curve.json", [])
    if summary.get("status") != "COMPLETE" or len(curve) < 10:
        return None
    if any(point.get("valid") != point.get("expected") or
           point.get("episodes") != point.get("expected") for point in curve):
        return None
    if not math.isfinite(float(summary.get("late3", float("nan")))):
        return None
    return summary


def run_evaluation(root: Path, plan: list[core.Cell]) -> None:
    for index, cell in enumerate(plan, 1):
        if not complete(root, cell):
            continue
        if valid_evaluation(root, cell) is not None:
            print(f"[EVAL SKIP {index}/{len(plan)}] {cell.key}", flush=True)
            continue
        print(f"[EVAL {index}/{len(plan)}] {cell.key}", flush=True)
        try:
            previous.evaluate_sparse(root, cell, {})
            if valid_evaluation(root, cell) is None:
                raise RuntimeError("稀疏曲线缺少完整有效的检查点回放")
        except Exception as exc:
            log_failure(root, cell, "evaluate", exc)
            print(f"[EVAL FAILED] {cell.key}: {exc!r}；继续下一格", flush=True)


def summarize(root: Path, plan: list[core.Cell]) -> None:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_cell: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], float] = {}
    for cell in plan:
        summary = valid_evaluation(root, cell)
        if summary is None:
            continue
        late = float(summary["late3"])
        groups[(cell.stage, cell.method)].append(late)
        by_key[(cell.key, cell.seed)] = late
        per_cell.append({"cell": cell.key, "train_seed": cell.seed,
                         "best": summary["best"], "late3": late,
                         "final": summary["final"],
                         "best_to_late_ratio": (late - float(summary["best"])) /
                         max(float(summary["best"]), 1e-9)})
    group_rows = []
    for (stage, method), values in sorted(groups.items()):
        group_rows.append({"stage": stage, "method": method, "n_train_seeds": len(values),
                           "late3_mean": statistics.mean(values),
                           "late3_sd": statistics.stdev(values) if len(values) > 1 else None,
                           "late3_cv": (statistics.stdev(values) / statistics.mean(values))
                           if len(values) > 1 else None,
                           "late3_worst": max(values)})
    pairs = []
    for method in ("S_TDM_EVENT_FUSION", "A_ICM_AC", "SA_TDMFUSION_ICM"):
        for seed in range(501, 506):
            left = by_key.get((f"K0__J4__{method}__s{seed}", seed))
            right = by_key.get((f"K1__J4__{method}__s{seed}", seed))
            if left is not None and right is not None:
                pairs.append({"comparison": "KL_GUARD_MINUS_500M_PPO", "method": method,
                              "seed": seed, "control_late3": left,
                              "treatment_late3": right, "delta": right - left})
    for seed in range(511, 515):
        left = by_key.get((f"C0__J4__S_TDM_EVENT_FUSION__s{seed}", seed))
        right = by_key.get((f"C1__J4__J4C_S__s{seed}", seed))
        if left is not None and right is not None:
            pairs.append({"comparison": "J4C_S_MINUS_J4_S", "method": "S_TDM_EVENT_FUSION",
                          "seed": seed, "control_late3": left,
                          "treatment_late3": right, "delta": right - left})
    completed_steps = sum(cell.steps for cell in plan if complete(root, cell))
    core.write_json(root / "supplement70m_summary.json", {
        "planned_training_steps": sum(cell.steps for cell in plan),
        "completed_nominal_steps": completed_steps,
        "completed_cells": sum(complete(root, cell) for cell in plan),
        "planned_cells": len(plan),
        "evaluation_boundary": "TRAIN_TRACE_DIAGNOSTIC_ONLY",
        "negative_paired_delta_means_improvement": True,
        "groups": group_rows, "pairs": pairs, "cells": per_cell,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="只运行一轮技术自检，不占正式预算")
    parser.add_argument("--resume", action="store_true", help="核对 manifest 后续跑未完成格")
    parser.add_argument("--train-only", action="store_true", help="只完成训练，暂不回放")
    parser.add_argument("--eval-only", action="store_true", help="训练完成后独立回放")
    args = parser.parse_args()
    if args.train_only and args.eval_only:
        parser.error("--train-only 和 --eval-only 不能同时使用")
    plan = make_plan()
    if args.plan_only:
        phases = {stage: sum(cell.steps for cell in plan if cell.stage == stage)
                  for stage in ("K0", "K1", "C0", "C1")}
        print(json.dumps({"cells": len(plan), "phase_steps": phases,
                          "total_steps": sum(phases.values()), "rollout": ROLLOUT,
                          "paired_kl_seeds": 5, "paired_j4c_seeds": 4},
                         ensure_ascii=False, indent=2))
        return
    if not args.eval_only and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请检查所选 GPU 和 PyTorch，不自动退回 CPU")
    core.methods.install_hooks()
    # 上一行注册的是 500M 方法；下面在其外层仅加入单因素 KL 处理。
    original_builder = core.v3.build_model

    def with_kl(*, env: Any, env_key: str, method_id: str, profile: Any,
                seed: int, run_dir: Path, device: str) -> Any:
        model = original_builder(env=env, env_key=env_key, method_id=method_id,
                                 profile=profile, seed=seed, run_dir=run_dir, device=device)
        model.target_kl = stage_kl(run_dir.parent.name)
        return model

    core.v3.build_model = with_kl
    freeze, profile, config = previous.prepare_configuration(None)
    assert not freeze and config["source"] == "PROJECT_S3_DEFAULT"
    trace_contract = verify_trace(Path(config["demand"]))
    root = args.run_dir.resolve()
    if args.smoke:
        smoke_cell = core.Cell("K1", "S_TDM_EVENT_FUSION", "J4", 999, ROLLOUT)
        smoke_root = root.with_name(root.name + "_technical_smoke")
        verify_saved_kl(smoke_root, smoke_cell)
        core.run_cell(smoke_root, smoke_cell, {}, profile, args.device, retry_failed=True)
        print("[SMOKE PASS] 一轮训练完成；正式预算尚未开始", flush=True)
        return
    if args.eval_only and not (root / "supplement70m_manifest.json").is_file():
        raise FileNotFoundError("找不到本补训的 manifest，不能对未启动的目录执行 --eval-only")
    manifest = manifest_for(plan, profile, config, trace_contract)
    ensure_manifest(root, manifest, resume=args.resume or args.eval_only)
    print(f"[START] 38 格，预算 {sum(cell.steps for cell in plan):,} 步；"
          f"训练表核对={trace_contract['match']}", flush=True)
    try:
        if not args.eval_only:
            run_training(root, plan, profile, args.device)
        if not args.train_only:
            run_evaluation(root, plan)
    finally:
        summarize(root, plan)
    print(f"[FINISH] 进度见 {root / 'supplement70m_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
