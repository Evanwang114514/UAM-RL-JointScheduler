# -*- coding: utf-8 -*-
"""一条命令连续完成约五亿步 UAM S/AC/S+AC 与联合控制实验。

沿用现有 S3/T2/V5 物理、ATT 奖励、PPO 和 50k 成对检查点。
不会等待人工晋级，也不以吞吐或验证成绩作为停训条件。
单组失败记录后尝试恢复或用预先声明的对照组补足；系统性故障不会伪装成有效训练。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

import train_uam_500m_stagewise as core


ROOT = Path(__file__).resolve().parent
ROLLOUT = 20_480
TARGET_STEPS = 499_998_720
DEFAULT_RUN_DIR = ROOT / "serial_runs" / "uam500m_continuous_s3"


def create_plan() -> list[core.Cell]:
    """固定预算与先后次序；每一格都能独立续训。"""
    plan: list[core.Cell] = []

    anchors = [
        ("UAGMC_SOURCE", "S3"),
        ("CURRENT", "J4"),
        ("SHARED", "J4"),
        ("S_TDM_EVENT_FUSION", "J4"),
        ("A_ICM_AC", "J4"),
        ("SA_TDMFUSION_ICM", "J4"),
    ]
    for method, env in anchors:
        for seed in range(401, 405):
            plan.append(core.Cell("A", method, env, seed, 60 * ROLLOUT))

    repairs = [
        "S_TDM_EVENT_FUSION", "S_EMA", "S_MASKED_EVENT_EMA",
        "A_ICM_AC", "AC_RELEVANT_ICM",
        "SA_TDMFUSION_ICM", "SA_ALTERNATE", "SA_EMA_RELEVANT", "SA_MASKED_RELEVANT",
    ]
    for method in repairs:
        for seed in range(405, 411):
            plan.append(core.Cell("B", method, "J4", seed, 100 * ROLLOUT))

    joint_lines = [
        ("S_TDM_EVENT_FUSION", "J4C_S"),
        ("A_ICM_AC", "J4C_AC"),
        ("SA_TDMFUSION_ICM", "J4C_SA"),
    ]
    for base, conditional in joint_lines:
        for env in ("J0", "J3", "J4"):
            for seed in range(411, 417):
                plan.append(core.Cell("C", base, env, seed, 100 * ROLLOUT))
        for seed in range(411, 417):
            plan.append(core.Cell("C", conditional, "J4", seed, 100 * ROLLOUT))

    long_run = [
        "SHARED", "S_TDM_EVENT_FUSION", "A_ICM_AC", "SA_TDMFUSION_ICM",
        "S_EMA", "AC_RELEVANT_ICM", "SA_EMA_RELEVANT", "J4C_SA",
    ]
    for method in long_run:
        for seed in range(417, 425):
            plan.append(core.Cell("D", method, "J4", seed, 150 * ROLLOUT))

    fresh = [
        "S_TDM_EVENT_FUSION", "A_ICM_AC", "SA_TDMFUSION_ICM",
        "S_EMA", "SA_EMA_RELEVANT", "J4C_SA",
    ]
    for method in fresh:
        for seed in range(425, 428):
            plan.append(core.Cell("E", method, "J4", seed, 43 * ROLLOUT))

    assert len(plan) == 232
    assert len({cell.key for cell in plan}) == len(plan)
    assert sum(cell.steps for cell in plan) == TARGET_STEPS
    return plan


def choose_freeze(explicit: Path | None) -> Path | None:
    """优先使用明确指定的冻结结果；否则寻找现有 50M 完整输出。"""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"指定的冻结结果不存在：{path}")
        return path
    direct = ROOT / "PRE500M_FINAL_FREEZE.json"
    if direct.is_file():
        return direct
    found = list((ROOT / "serial_runs").glob("**/PRE500M_FINAL_FREEZE.json"))
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def prepare_configuration(freeze_path: Path | None) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """不另设启动闸门，但真实训练物理与 trace 必须可读取。"""
    if freeze_path is None:
        trace = core.v3.TRAIN_FILE.resolve()
        if not trace.is_file():
            raise FileNotFoundError(f"项目基准乘客文件缺失：{trace}")
        profile = core.v3.SpeedProfile("FIXED_10x2048_b1024", 10, 2048, 1024)
        return {}, profile, {
            "source": "PROJECT_S3_DEFAULT", "physics": "S3", "topology": "T2",
            "fleet": 40, "demand": str(trace), "demand_sha256": core.digest(trace),
            "n_envs": 10, "n_steps": 2048, "batch_size": 1024,
            "gamma": 1.0, "global_rollout": ROLLOUT,
            "evaluation_boundary": "同一基准 trace 的回放只能用于调试，不能称为独立测试",
        }
    freeze = core.read_json(freeze_path)
    frozen_env = freeze.get("frozen_environment", {})
    ppo = freeze.get("frozen_ppo", {})
    demand = freeze.get("demand_protocol", {})
    if (frozen_env.get("physics"), frozen_env.get("topology"), frozen_env.get("fleet_size")) != ("S3", "T2", 40):
        raise RuntimeError("冻结文件不是已约定的 S3/T2/40 架物理环境")
    if int(ppo.get("n_envs", 0)) * int(ppo.get("n_steps", 0)) != ROLLOUT:
        raise RuntimeError("冻结文件的全局 rollout 不是 20480")
    if float(ppo.get("gamma", -1)) != 1.0:
        raise RuntimeError("冻结文件的 gamma 不是 1.0")
    if not demand.get("selected_protocol") or not demand.get("train_trace_bank"):
        raise RuntimeError("冻结文件缺少已选训练 trace 协议")
    freeze["_sha256"] = core.digest(freeze_path)
    profile = core.profile_from_freeze(freeze)
    for seed in (401, 405, 411, 417, 425, 427):
        core.choose_trace(freeze, seed)
    return freeze, profile, {
        "source": "PRE500M_FINAL_FREEZE", "freeze_file": str(freeze_path),
        "freeze_sha256": freeze["_sha256"], "physics": "S3", "topology": "T2",
        "fleet": 40, "selected_load": frozen_env.get("selected_load"),
        "demand_protocol": demand["selected_protocol"],
        "n_envs": profile.n_envs, "n_steps": profile.n_steps,
        "batch_size": profile.batch_size, "gamma": 1.0,
        "global_rollout": ROLLOUT,
        "evaluation_boundary": "F5 参与选择的 heldout traces 仅作为 validation",
    }


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        ).strip()
    except Exception:
        return "UNAVAILABLE"


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def fallback_cell(target: core.Cell, index: int, attempt: int) -> core.Cell:
    """原格无法训练时，用预先声明的强对照补足同样步数。"""
    fallbacks = [
        ("S_TDM_EVENT_FUSION", "J4"),
        ("A_ICM_AC", "J4"),
        ("SA_TDMFUSION_ICM", "J4"),
        ("SHARED", "J4"),
        ("CURRENT", "J4"),
        ("UAGMC_SOURCE", "S3"),
    ]
    method, env = fallbacks[attempt % len(fallbacks)]
    return core.Cell(target.stage, method, env, 900_000 + 10 * index + attempt,
                     target.steps)


def completed_record(root: Path, cell: core.Cell) -> bool:
    return core.read_json(core.cell_dir(root, cell) / "run_end.json", {}).get("status") == "COMPLETE"


def run_slot(root: Path, target: core.Cell, index: int, freeze: dict[str, Any],
             profile: Any, device: str) -> bool:
    """每个名额只认完整训练，重启时自动识别原格与替补格。"""
    slot_path = root / "slots" / f"{index:04d}.json"
    previous = core.read_json(slot_path, {})
    if previous.get("status") == "COMPLETE":
        return True
    candidates = [target] + [fallback_cell(target, index, j) for j in range(6)]
    for candidate in candidates:
        if completed_record(root, candidate):
            core.write_json(slot_path, {
                "status": "COMPLETE", "requested": asdict(target),
                "trained": asdict(candidate), "replacement": candidate != target,
            })
            return True

    failures = []
    for candidate_index, candidate in enumerate(candidates):
        # 原方法先尝试两次；替补按预注册顺序各尝试一次。
        max_attempts = 2 if candidate_index == 0 else 1
        for trial in range(max_attempts):
            print(f"[SLOT {index + 1}/232] {candidate.key} | 尝试 {trial + 1}/{max_attempts}", flush=True)
            try:
                core.run_cell(root, candidate, freeze, profile, device, retry_failed=True)
                if completed_record(root, candidate):
                    core.write_json(slot_path, {
                        "status": "COMPLETE", "requested": asdict(target),
                        "trained": asdict(candidate), "replacement": candidate != target,
                    })
                    return True
                raise RuntimeError("训练调用返回但未生成 COMPLETE 记录")
            except Exception as exc:
                info = {"slot": index, "candidate": asdict(candidate),
                        "trial": trial + 1, "error": repr(exc),
                        "traceback": traceback.format_exc(), "time": time.time()}
                append_event(root / "failures.jsonl", info)
                failures.append(info)
                print(f"[FAILED] {candidate.key}: {exc!r}；继续下一次尝试", flush=True)
    core.write_json(slot_path, {
        "status": "FAILED", "requested": asdict(target),
        "attempted": [asdict(cell) for cell in candidates],
        "last_error": failures[-1]["error"] if failures else None,
    })
    return False


def completed_slots(root: Path, count: int) -> int:
    return sum(core.read_json(root / "slots" / f"{i:04d}.json", {}).get("status") == "COMPLETE"
               for i in range(count))


def validation_traces(freeze: dict[str, Any]) -> tuple[list[Path], str]:
    if freeze:
        paths = [Path(p) for p in freeze["demand_protocol"].get("heldout_trace_bank", [])]
        paths = [p for p in paths if p.is_file()]
        if paths:
            return paths[:3], "F5_VALIDATION_NOT_FINAL_TEST"
    return [core.v3.TRAIN_FILE.resolve()], "TRAIN_TRACE_DIAGNOSTIC_ONLY"


def evaluate_sparse(root: Path, cell: core.Cell, freeze: dict[str, Any]) -> None:
    """训练全部结束后稀疏回放，保留 50k 原始检查点供后续完整审计。"""
    analysis = core.cell_dir(root, cell) / "analysis"
    if core.read_json(analysis / "sparse_summary.json", {}).get("status") == "COMPLETE":
        return
    checkpoints = []
    for model_path in (core.cell_dir(root, cell) / "checkpoints").glob("uam_ppo_*_steps.zip"):
        try:
            step = int(model_path.stem.split("_")[2])
        except (IndexError, ValueError):
            continue
        vec_path = model_path.with_name(f"uam_ppo_vecnormalize_{step}_steps.pkl")
        if vec_path.is_file():
            checkpoints.append((step, model_path, vec_path))
    checkpoints.sort(key=lambda triple: triple[0])
    if not checkpoints:
        final_model = core.cell_dir(root, cell) / "final_model.zip"
        final_vec = core.cell_dir(root, cell) / "final_vecnormalize.pkl"
        if not final_model.is_file() or not final_vec.is_file():
            raise RuntimeError(f"缺少成对检查点：{cell.key}")
        completed = core.read_json(core.cell_dir(root, cell) / "run_end.json", {})
        checkpoints = [(int(completed.get("actual_steps", cell.steps)), final_model, final_vec)]
    indices = sorted(set(np.linspace(0, len(checkpoints) - 1,
                                     min(16, len(checkpoints))).round().astype(int)))
    traces, boundary = validation_traces(freeze)
    rows = []
    for i in indices:
        step, model_path, vec_path = checkpoints[i]
        for trace in traces:
            core.pre80.set_trace(trace)
            for seed in (123, 124):
                try:
                    row = core.v3.evaluate_checkpoint(
                        env_key=cell.env, method_id=cell.method, model_path=model_path,
                        vec_path=vec_path, train_step=step, eval_seed=seed,
                        run_dir=core.cell_dir(root, cell), max_time=2500,
                    )
                    row.update({"validation_trace": str(trace), "boundary": boundary,
                                "train_seed": cell.seed})
                    rows.append(row)
                except Exception as exc:
                    append_event(root / "eval_failures.jsonl", {
                        "cell": cell.key, "step": step, "trace": str(trace),
                        "seed": seed, "error": repr(exc), "time": time.time(),
                    })
    analysis.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({field for row in rows for field in row})
        with (analysis / "sparse_curve_raw.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    curve = []
    expected_episodes = len(traces) * 2
    for step in sorted({int(row["train_step"]) for row in rows}):
        grouped = [row for row in rows if int(row["train_step"]) == step]
        valid = [row for row in grouped if row.get("valid_full_completion")
                 and math.isfinite(float(row.get("ATT", float("nan"))))]
        curve.append({"step": step, "episodes": len(grouped), "expected": expected_episodes,
                      "valid": len(valid),
                      "ATT": float(np.mean([float(row["ATT"]) for row in valid])) if valid else None})
    valid_att = [item["ATT"] for item in curve if item["ATT"] is not None
                 and item["valid"] == item["expected"]]
    core.write_json(analysis / "sparse_curve.json", curve)
    core.write_json(analysis / "sparse_summary.json", {
        "status": "COMPLETE" if valid_att else "NO_FULL_COMPLETION",
        "evaluation_boundary": boundary, "n_checkpoints": len(curve),
        "best": min(valid_att) if valid_att else None,
        "late3": float(np.mean(valid_att[-3:])) if valid_att else None,
        "final": valid_att[-1] if valid_att else None,
        "training_replicate": cell.seed,
        "evaluation_traces_are_not_independent_training_samples": True,
    })


def summarize(root: Path, plan: list[core.Cell]) -> None:
    groups: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    replacement_count = 0
    for i, requested in enumerate(plan):
        slot = core.read_json(root / "slots" / f"{i:04d}.json", {})
        if slot.get("status") != "COMPLETE":
            continue
        trained = core.Cell(**slot["trained"])
        replacement_count += bool(slot.get("replacement"))
        summary = core.read_json(core.cell_dir(root, trained) / "analysis" / "sparse_summary.json", {})
        if summary.get("status") == "COMPLETE":
            groups[(trained.stage, trained.method, trained.env, trained.steps)].append(float(summary["late3"]))
    rows = []
    for (stage, method, env, steps), vals in sorted(groups.items()):
        array = np.asarray(vals, dtype=float)
        rows.append({"stage": stage, "method": method, "env": env,
                     "steps_per_model": steps, "n_trained_models": len(vals),
                     "late3_mean": float(array.mean()),
                     "late3_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
                     "late3_worst": float(array.max())})
    actual_spent = 0
    for path in root.glob("*/**/run_end.json"):
        if path.parent.parent.name in {"A", "B", "C", "D", "E"}:
            actual_spent += int(core.read_json(path, {}).get("spent_steps", 0))
    core.write_json(root / "one_shot_aggregate.json", {
        "completed_slots": completed_slots(root, len(plan)),
        "nominal_completed_training_steps": sum(
            requested.steps for i, requested in enumerate(plan)
            if core.read_json(root / "slots" / f"{i:04d}.json", {}).get("status") == "COMPLETE"),
        "target_training_steps": TARGET_STEPS,
        "actual_consumed_steps_including_failed_retries": actual_spent,
        "replacement_slots": replacement_count,
        "methods": rows,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--freeze", type=Path, default=None,
                        help="可选；默认自动寻找 50M 最终冻结文件，找不到则用项目 S3 基准配置")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="可选的一轮技术自检，不影响正式五亿步")
    parser.add_argument("--train-only", action="store_true", help="五亿步结束后暂不自动回放")
    parser.add_argument("--eval-only", action="store_true", help="只接着做训练后的稀疏回放")
    args = parser.parse_args()

    plan = create_plan()
    if args.plan_only:
        phases = {phase: sum(cell.steps for cell in plan if cell.stage == phase)
                  for phase in "ABCDE"}
        print(json.dumps({"cells": len(plan), "phase_steps": phases,
                          "total_steps": sum(phases.values()), "rollout": ROLLOUT},
                         ensure_ascii=False, indent=2))
        return
    if not args.eval_only and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 未识别 CUDA；请先检查环境或明确传入 --device cpu")

    core.methods.install_hooks()
    root = args.run_dir.resolve()
    old_manifest = core.read_json(root / "one_shot_manifest.json", {})
    old_config = old_manifest.get("frozen_configuration", {})
    if args.freeze is None and old_config.get("source") == "PROJECT_S3_DEFAULT":
        freeze_path = None
    elif args.freeze is None and old_config.get("source") == "PRE500M_FINAL_FREEZE":
        freeze_path = Path(old_config["freeze_file"])
    else:
        freeze_path = choose_freeze(args.freeze)
    freeze, profile, config = prepare_configuration(freeze_path)
    if args.smoke:
        check = core.Cell("SMOKE", "SA_EMA_RELEVANT", "J4", 999, ROLLOUT)
        result = core.run_cell(root / "_technical_smoke", check, freeze, profile,
                               args.device, retry_failed=True)
        print(f"[SMOKE] {result}", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"script": Path(__file__).name, "git_commit": current_commit(),
                "frozen_configuration": config, "plan": [asdict(cell) for cell in plan],
                "target_steps": TARGET_STEPS,
                "selection": "预注册固定顺序；无人工晋级、无性能早停、无吞吐闸门"}
    manifest_path = root / "one_shot_manifest.json"
    existing = core.read_json(manifest_path, None)
    if existing is not None:
        # 已有目录中的运行定义不得因脚本或参数变化而悄悄混入新结果。
        comparable = {key: value for key, value in existing.items() if key != "git_commit"}
        current = {key: value for key, value in manifest.items() if key != "git_commit"}
        if comparable != current:
            raise RuntimeError("现有运行目录与本次固定计划不同；换新 --run-dir 或恢复原配置")
    else:
        core.write_json(manifest_path, manifest)

    print(f"[START] {len(plan)} 格，总目标 {TARGET_STEPS:,} 步", flush=True)
    print(f"[CONFIG] {json.dumps(config, ensure_ascii=False)}", flush=True)
    if not args.eval_only:
        consecutive_unrecoverable = 0
        for i, cell in enumerate(plan):
            ok = run_slot(root, cell, i, freeze, profile, args.device)
            done = completed_slots(root, len(plan))
            consecutive_unrecoverable = 0 if ok else consecutive_unrecoverable + 1
            print(f"[PROGRESS] {done}/{len(plan)} 格完成；刚才{'完成' if ok else '全部失败'}：{cell.key}",
                  flush=True)
            if consecutive_unrecoverable >= 3:
                summarize(root, plan)
                raise RuntimeError("连续三个名额连全部替补都无法运行，属于系统性故障；保留检查点并停止空耗")
        failed = [i for i in range(len(plan)) if core.read_json(
            root / "slots" / f"{i:04d}.json", {}).get("status") != "COMPLETE"]
        if failed:
            # 单组失败不阻断后续训练；整轮结束时再尝试一次未完成名额。
            print(f"[REVISIT] {len(failed)} 个失败名额再试一次", flush=True)
            for i in failed:
                run_slot(root, plan[i], i, freeze, profile, args.device)
        if completed_slots(root, len(plan)) != len(plan):
            summarize(root, plan)
            raise RuntimeError("部分方法与全部替补均失败；已保存所有检查点，同一命令可续训，不能伪报五亿完成")

    if not args.train_only:
        for i, requested in enumerate(plan):
            slot = core.read_json(root / "slots" / f"{i:04d}.json", {})
            if slot.get("status") != "COMPLETE":
                continue
            trained = core.Cell(**slot["trained"])
            try:
                evaluate_sparse(root, trained, freeze)
            except Exception as exc:
                append_event(root / "eval_failures.jsonl", {
                    "cell": trained.key, "error": repr(exc),
                    "traceback": traceback.format_exc(), "time": time.time(),
                })
                print(f"[EVAL FAILED] {trained.key}: {exc!r}；继续下一组", flush=True)
    summarize(root, plan)
    print(f"[FINISH] 有效完成 {completed_slots(root, len(plan))}/{len(plan)} 格；"
          f"名义训练预算 {TARGET_STEPS:,} 步；结果目录 {root}", flush=True)


if __name__ == "__main__":
    main()
