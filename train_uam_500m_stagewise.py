# -*- coding: utf-8 -*-
"""500M 预算上限的分阶段 UAM 联合控制实验编排器。

先运行 --plan-only 或 --smoke。正式阶段必须提供完整的 50M 冻结文件。
训练与回放分开执行；每 50k 保存模型和归一化器，失败不会吞掉证据。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

import train_uam_pre5b_80m_autofunnel as pre80
import uam500m_methods as methods

v3 = pre80.v3
v4 = pre80.v4
ROOT = Path(__file__).resolve().parent
CAP = 500_000_000
PHASE_CAPS = {"A": 30_000_000, "B": 105_000_000,
              "C": 85_000_000, "D": 240_000_000, "RESERVE": 40_000_000}
assert sum(PHASE_CAPS.values()) == CAP
ROLLOUT = 20_480
SAVE_EVERY = 50_000
EVAL_SEEDS = (123, 124)


@dataclass(frozen=True)
class Cell:
    stage: str
    method: str
    env: str
    seed: int
    steps: int

    @property
    def key(self) -> str:
        return f"{self.stage}__{self.env}__{self.method}__s{self.seed}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_freeze(path: Path, *, strict: bool) -> dict[str, Any]:
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"正式训练必须先完成 F3–F8：{path}")
        return {}
    data = read_json(path)
    ppo = data.get("frozen_ppo", {})
    physics = data.get("frozen_environment", {})
    demand = data.get("demand_protocol", {})
    joint = data.get("joint_contract", {})
    if ppo.get("global_rollout") != ROLLOUT or float(ppo.get("gamma", -1)) != 1.0:
        raise RuntimeError("冻结文件的 rollout/gamma 与既定合同不一致")
    if (physics.get("physics"), physics.get("topology"), physics.get("fleet_size")) != ("S3", "T2", 40):
        raise RuntimeError("冻结文件不是 S3/T2/fleet40 物理环境")
    if not isinstance(physics.get("selected_load"), (int, float)):
        raise RuntimeError("F3 未冻结 main load")
    if not demand.get("selected_protocol") or not demand.get("train_trace_bank"):
        raise RuntimeError("F5 未冻结 demand protocol / trace bank")
    if joint.get("physical_semantics") != "V5_CLEAN_MINIMAL_REPOSITION":
        raise RuntimeError("F6 未确认 V5 clean 物理语义")
    if not data.get("formal_single_joint") or not data.get("omitted_longrun_confirmation"):
        raise RuntimeError("F7/F8 尚无完整结果，不能开始 500M")
    if int(ppo.get("n_envs", 0)) * int(ppo.get("n_steps", 0)) != ROLLOUT:
        raise RuntimeError("n_envs × n_steps 必须等于 20480")
    if int(ppo.get("batch_size", 0)) not in (1024, 4096):
        raise RuntimeError("batch 必须来自 F2 的冻结候选")
    return data


def profile_from_freeze(freeze: dict[str, Any]) -> Any:
    p = freeze["frozen_ppo"]
    return v3.SpeedProfile(str(p["profile_name"]), int(p["n_envs"]),
                           int(p["n_steps"]), int(p["batch_size"]))


def planned_cells(stage: str, selected: dict[str, Any]) -> list[Cell]:
    if stage == "A":
        anchors = [("UAGMC_SOURCE", "S3"), ("CURRENT", "J4"),
                   ("SHARED", "J4"), ("S_TDM_EVENT_FUSION", "J4"),
                   ("A_ICM_AC", "J4"), ("SA_TDMFUSION_ICM", "J4")]
        return [Cell(stage, m, e, s, 1_228_800) for m, e in anchors for s in range(401, 405)]
    if stage == "B1":
        variants = ("S_EMA", "S_MASKED_EVENT_EMA", "AC_RELEVANT_ICM",
                    "SA_ALTERNATE", "SA_EMA_RELEVANT", "SA_MASKED_RELEVANT")
        return [Cell(stage, m, "J4", s, 409_600) for m in variants for s in range(401, 404)]
    if stage == "C1":
        lines = (("S_TDM_EVENT_FUSION", "J4C_S"),
                 ("A_ICM_AC", "J4C_AC"),
                 ("SA_TDMFUSION_ICM", "J4C_SA"))
        out = []
        for base, conditional in lines:
            for env in ("J0", "J3", "J4"):
                out.extend(Cell(stage, base, env, s, 409_600) for s in range(401, 404))
            out.extend(Cell(stage, conditional, "J4", s, 409_600) for s in range(401, 404))
        return out
    if stage in {"B2", "B3", "C2", "C3", "D", "D2"}:
        picks = selected.get(stage, [])
        if not picks:
            raise RuntimeError(f"{stage} 需要选拔文件中的非空 {stage} 数组")
        limit = {"B2": 5, "B3": 2, "C2": 3, "C3": 1, "D": 3, "D2": 1}[stage]
        if len(picks) > limit:
            raise RuntimeError(f"{stage} 最多 {limit} 个候选")
        allocation = {
            "B2": (501, 506, 1_228_800), "B3": (551, 555, 2_457_600),
            "C2": (501, 506, 1_228_800), "C3": (551, 556, 2_048_000),
            "D": (601, 611, 3_072_000), "D2": (701, 705, 3_072_000),
        }
        start, stop, steps = allocation[stage]
        pairs = [(str(p["method"]).upper(), str(p["env"]).upper()) for p in picks]
        if stage == "B2":
            pairs += [("S_TDM_EVENT_FUSION", "J4"), ("A_ICM_AC", "J4"),
                      ("SA_TDMFUSION_ICM", "J4")]
        elif stage == "B3":
            pairs += [matched_reference(m, e) for m, e in pairs]
        elif stage in {"C2", "C3"}:
            pairs += [matched_reference(m, e) for m, e in pairs]
        elif stage == "D":
            pairs += [("UAGMC_SOURCE", "S3"), ("CURRENT", "J4"), ("SHARED", "J4")]
            pairs += [matched_reference(m, e) for m, e in pairs[:len(picks)]]
            if len(set(pairs)) > 7:
                raise RuntimeError("D 阶段候选及其匹配对照超过 7 个；缩减 finalist")
        else:
            pairs += [matched_reference(m, e) for m, e in pairs]
        return [Cell(stage, m, e, s, steps) for m, e in dict.fromkeys(pairs)
                for s in range(start, stop)]
    raise ValueError(stage)


def matched_reference(method: str, env: str) -> tuple[str, str]:
    """晋级时自动携带同架构、同 seed、同训练长度的对照。"""
    if method.startswith("J4C_"):
        base = {"J4C_S": "S_TDM_EVENT_FUSION", "J4C_AC": "A_ICM_AC",
                "J4C_SA": "SA_TDMFUSION_ICM"}[method]
        return base, "J4"
    if env in {"J0", "J3"}:
        return "CURRENT", env
    if method.startswith("S_"):
        return "S_TDM_EVENT_FUSION", env
    if method.startswith("AC_"):
        return "A_ICM_AC", env
    if method.startswith("SA_"):
        return "SA_TDMFUSION_ICM", env
    return "CURRENT", env


def phase_of(stage: str) -> str:
    return stage[0]


def assert_design(cells: list[Cell], stage: str) -> None:
    known = {"UAGMC_SOURCE", "CURRENT", "SHARED", "S_TDM_EVENT_FUSION",
             "A_ICM_AC", "SA_TDMFUSION_ICM"} | set(methods.VARIANT_COMPONENTS)
    if len({c.key for c in cells}) != len(cells):
        raise RuntimeError("重复 cell")
    for c in cells:
        if c.method not in known or c.env not in {"S3", "J0", "J3", "J4"}:
            raise RuntimeError(f"未知方法或环境：{c}")
        if c.method.startswith("J4C_") and c.env != "J4":
            raise RuntimeError("条件式 J4 只能在 J4 环境使用")
        if c.steps % ROLLOUT:
            raise RuntimeError("训练长度必须整除全局 rollout")
    if sum(c.steps for c in cells) > PHASE_CAPS[phase_of(stage)]:
        raise RuntimeError("阶段计划超出预算上限")


def choose_trace(freeze: dict[str, Any], seed: int) -> Path:
    demand = freeze["demand_protocol"]
    if str(demand["selected_protocol"]).upper().startswith("MULTI"):
        bank = list(demand["train_trace_bank"])
        trace = bank[(int(seed) - 1) % len(bank)]
    else:
        trace = freeze["frozen_environment"]["selected_trace"]
    path = Path(trace).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"冻结 demand trace 不在当前机器：{path}")
    return path.resolve()


def selected_validation_traces(freeze: dict[str, Any]) -> list[Path]:
    # F5 使用过的所谓 test traces 一律改称 validation，不碰真正最终测试集。
    traces = [Path(p) for p in freeze["demand_protocol"]["heldout_trace_bank"]]
    for p in traces:
        if not p.exists():
            raise FileNotFoundError(f"validation trace 缺失：{p}")
    return traces


def lock_final_test_bank(manifest_path: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    """只核验最终测试集的身份，不在选模阶段读取或回放其乘客内容。"""
    if not manifest_path.exists():
        raise FileNotFoundError(f"正式训练前须预注册 untouched FINAL_TEST bank：{manifest_path}")
    manifest = read_json(manifest_path)
    entries = manifest.get("traces", [])
    if not entries or not isinstance(entries, list):
        raise RuntimeError("FINAL_TEST bank 清单至少包含一个 trace")
    known_paths = {str(Path(p).resolve()).lower() for p in freeze["demand_protocol"]["train_trace_bank"]}
    known_paths.update(str(p.resolve()).lower() for p in selected_validation_traces(freeze))
    known_paths.add(str(Path(freeze["frozen_environment"]["selected_trace"]).resolve()).lower())
    locked = []
    for entry in entries:
        p = Path(entry["path"]).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"FINAL_TEST trace 缺失：{p}")
        if str(p).lower() in known_paths:
            raise RuntimeError(f"FINAL_TEST 与训练/验证 trace 路径重合：{p}")
        actual_hash = digest(p)
        if actual_hash != str(entry["sha256"]).lower():
            raise RuntimeError(f"FINAL_TEST trace 哈希不一致：{p}")
        locked.append({"path": str(p), "sha256": actual_hash})
    if len({x["sha256"] for x in locked}) != len(locked):
        raise RuntimeError("FINAL_TEST bank 内有重复文件")
    return {"manifest_sha256": digest(manifest_path), "traces": locked,
            "use_before_formal_method_selection": False}


def cell_dir(root: Path, c: Cell) -> Path:
    return root / c.stage / c.key


def all_spent(root: Path) -> tuple[int, dict[str, int]]:
    spent, phases = 0, {p: 0 for p in "ABCD"}
    for manifest in root.glob("*/**/cell_manifest.json"):
        run_dir = manifest.parent
        data = read_json(run_dir / "run_end.json", {})
        checkpoint = latest_checkpoint(run_dir)
        # 进程被强制中断时可能没有结束记录，但已落盘步数仍须计入预算。
        steps = max(int(data.get("spent_steps", 0)), checkpoint[0] if checkpoint else 0)
        stage = run_dir.parent.name
        if stage and stage[0] in phases:
            spent += steps
            phases[stage[0]] += steps
    return spent, phases


def latest_checkpoint(run_dir: Path) -> tuple[int, Path, Path] | None:
    candidates = []
    for model_path in (run_dir / "checkpoints").glob("uam_ppo_*_steps.zip"):
        try:
            step = int(model_path.stem.split("_")[2])
        except (IndexError, ValueError):
            continue
        vec_path = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        if vec_path.exists():
            candidates.append((step, model_path, vec_path))
    return max(candidates, key=lambda x: x[0]) if candidates else None


def algo_for(method: str) -> type[PPO]:
    if method in methods.AUX_METHODS:
        return methods.StabilizedAuxPPO
    if v4.custom_method_needs_aux(method):
        return v4.DiscoveryAuxPPO
    return PPO


def run_cell(root: Path, c: Cell, freeze: dict[str, Any], profile: Any,
             device: str, retry_failed: bool, smoke: bool = False) -> dict[str, Any]:
    run_dir = cell_dir(root, c)
    run_dir.mkdir(parents=True, exist_ok=True)
    old = read_json(run_dir / "run_end.json", {})
    if old.get("status") == "COMPLETE":
        print(f"[SKIP] {c.key} 已完成", flush=True)
        return old
    if old.get("status") == "FAILED" and not retry_failed:
        print(f"[SKIP] {c.key} 上次失败；--retry-failed 才会重试", flush=True)
        return old
    trace = choose_trace(freeze, c.seed) if freeze else v3.TRAIN_FILE
    pre80.set_trace(trace)
    contract = {**asdict(c), "trace": str(trace), "profile": asdict(profile),
                "freeze_sha256": freeze.get("_sha256", "SMOKE_ONLY")}
    manifest_path = run_dir / "cell_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != contract:
        raise RuntimeError(f"已有 cell 定义不同，拒绝混训：{run_dir}")
    write_json(manifest_path, contract)
    v3.seed_all(c.seed)
    env = None
    model = None
    start_step = 0
    old_spent = int(old.get("spent_steps", 0))
    t0 = time.perf_counter()
    try:
        env = v3.build_vec_env(env_key=c.env, method_id=c.method, profile=profile,
                               seed=c.seed, run_dir=run_dir, max_time=2500)
        checkpoint = latest_checkpoint(run_dir)
        if checkpoint is not None:
            start_step, model_path, vec_path = checkpoint
            old_spent = max(old_spent, start_step)
            env = v3.TauPreservingVecNormalize.load(str(vec_path), env.venv)
            model = algo_for(c.method).load(str(model_path), env=env, device=device)
            print(f"[RESUME] {c.key} 从 {start_step:,} 步继续；环境轨迹重新开始，非 bitwise 续训", flush=True)
        else:
            model = v3.build_model(env=env, env_key=c.env, method_id=c.method,
                                   profile=profile, seed=c.seed, run_dir=run_dir, device=device)
        if start_step >= c.steps:
            raise RuntimeError("已有 checkpoint 已达到目标但无完成标记，需要人工核对")
        callbacks = [CheckpointCallback(
            save_freq=SAVE_EVERY // profile.n_envs,
            save_path=str(run_dir / "checkpoints"), name_prefix="uam_ppo",
            save_vecnormalize=True, verbose=0,
        )]
        if c.method in methods.RELEVANT_METHODS:
            callbacks.append(methods.RelevanceRecorder())
        model.learn(total_timesteps=c.steps - start_step,
                    callback=CallbackList(callbacks),
                    reset_num_timesteps=(start_step == 0), progress_bar=False)
        actual = int(model.num_timesteps)
        model.save(run_dir / "final_model.zip")
        env.save(run_dir / "final_vecnormalize.pkl")
        spent = old_spent + max(0, actual - start_step)
        elapsed = time.perf_counter() - t0
        result = {"status": "COMPLETE", "actual_steps": actual,
                  "spent_steps": spent, "elapsed_seconds": elapsed,
                  "train_sps": (actual - start_step) / max(elapsed, 1e-9)}
        write_json(run_dir / "run_end.json", result)
        return result
    except Exception as exc:
        actual = int(getattr(model, "num_timesteps", start_step))
        result = {"status": "FAILED", "actual_steps": actual,
                  "spent_steps": old_spent + max(0, actual - start_step),
                  "error_type": type(exc).__name__, "error": traceback.format_exc()}
        write_json(run_dir / "run_end.json", result)
        raise
    finally:
        if env is not None:
            env.close()
        try:
            v3.core.restore_process_patches()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_cell(run_dir: Path, c: Cell, freeze: dict[str, Any]) -> None:
    end = read_json(run_dir / "run_end.json", {})
    if end.get("status") != "COMPLETE":
        return
    analysis = run_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for checkpoint in sorted((run_dir / "checkpoints").glob("uam_ppo_*_steps.zip")):
        try:
            step = int(checkpoint.stem.split("_")[2])
        except (IndexError, ValueError):
            continue
        vec_path = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        if not vec_path.exists():
            continue
        for trace in selected_validation_traces(freeze):
            pre80.set_trace(trace)
            for seed in EVAL_SEEDS:
                row = v3.evaluate_checkpoint(
                    env_key=c.env, method_id=c.method, model_path=checkpoint,
                    vec_path=vec_path, train_step=step, eval_seed=seed,
                    run_dir=run_dir, max_time=2500,
                )
                row.update({"train_seed": c.seed, "validation_trace": str(trace)})
                rows.append(row)
    if not rows:
        raise RuntimeError(f"没有可评估的成对 checkpoint：{run_dir}")
    fields = sorted({k for row in rows for k in row})
    with (analysis / "validation_curve_raw.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    curve = []
    for step in sorted({int(r["train_step"]) for r in rows}):
        group = [r for r in rows if int(r["train_step"]) == step]
        valid = [r for r in group if bool(r.get("valid_full_completion"))
                 and math.isfinite(float(r.get("ATT", float("nan"))))]
        def mean_metric(metric: str, source: list[dict[str, Any]]) -> float:
            vals = [float(r.get(metric, float("nan"))) for r in source]
            vals = [x for x in vals if math.isfinite(x)]
            return float(np.mean(vals)) if vals else float("nan")
        curve.append({"step": step, "n_eval": len(group), "n_valid": len(valid),
                      "ATT": mean_metric("ATT", valid), "AGT": mean_metric("AGT", valid),
                      "AFT": mean_metric("AFT", valid), "AWT": mean_metric("AWT", valid),
                      "completion": mean_metric("completion_rate", group),
                      "dispatch_success": mean_metric("dispatch_success_rate", group),
                      "passenger_V0_share": mean_metric("passenger_V0_share", group),
                      "aircraft_V0_share": mean_metric("aircraft_V0_share", group)})
    write_json(analysis / "validation_curve.json", curve)
    att = [r["ATT"] for r in curve if math.isfinite(r["ATT"]) and r["n_valid"] == r["n_eval"]]
    if not att:
        write_json(analysis / "summary.json", {"status": "NO_FULL_COMPLETION"})
        return
    late = float(np.mean(att[-3:]))
    best = float(min(att))
    write_json(analysis / "summary.json", {
        "status": "VALID", "best": best, "late3": late, "final": float(att[-1]),
        "best_to_late_ratio": (late - best) / best,
        "late_slope": (att[-1] - att[-3]) / max(1, len(att[-3:]) - 1),
        "n_checkpoints": len(att), "training_replicate": c.seed,
        "late3_AGT": float(np.nanmean([r["AGT"] for r in curve[-3:]])),
        "late3_AFT": float(np.nanmean([r["AFT"] for r in curve[-3:]])),
        "late3_AWT": float(np.nanmean([r["AWT"] for r in curve[-3:]])),
        "late3_completion": float(np.nanmean([r["completion"] for r in curve[-3:]])),
        "validation_traces_are_not_independent_train_samples": True,
    })


def aggregate_stage(root: Path, cells: list[Cell], stage: str) -> None:
    """先汇总每个训练模型的 trace 均值，再跨训练 replicate 求稳定性。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in cells:
        summary = read_json(cell_dir(root, c) / "analysis" / "summary.json", {})
        if summary.get("status") == "VALID":
            groups.setdefault((c.method, c.env), []).append(summary)
    rows = []
    for (method, env), values in sorted(groups.items()):
        late = np.asarray([float(v["late3"]) for v in values], dtype=float)
        best_late = np.asarray([float(v["best_to_late_ratio"]) for v in values], dtype=float)
        rows.append({
            "stage": stage, "method": method, "env": env, "n_train_replicates": len(values),
            "late_mean": float(late.mean()), "late_sd": float(late.std(ddof=1)) if len(late)>1 else float("nan"),
            "late_cv": float(late.std(ddof=1)/late.mean()) if len(late)>1 and late.mean()>0 else float("nan"),
            "late_worst": float(late.max()), "late_range": float(late.max()-late.min()),
            "best_to_late_mean": float(best_late.mean()),
            "n_catastrophic_over_20pct": int((best_late>0.2).sum()),
            "statistical_unit": "trained_model_not_eval_trace",
        })
    write_json(root / stage / "stage_aggregate.json", rows)
    if rows:
        with (root / stage / "stage_aggregate.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("A", "B1", "B2", "B3", "C1", "C2", "C3", "D", "D2"), default="A")
    ap.add_argument("--freeze", type=Path, default=ROOT / "PRE500M_FINAL_FREEZE.json")
    ap.add_argument("--final-test-bank", type=Path, default=ROOT / "FINAL_TEST_BANK.json")
    ap.add_argument("--suite-root", type=Path, default=ROOT / "serial_runs" / "uam500m_stagewise_MAIN")
    ap.add_argument("--selection-file", type=Path)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--resume-suite", action="store_true", help="复用现有 suite；未完成 cell 从成对 checkpoint 继续")
    ap.add_argument("--resume-cell", type=str, default="", help="只处理指定 cell key")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--max-budget-steps", type=int, default=CAP)
    ap.add_argument("--min-train-sps", type=float, default=1000.0)
    ap.add_argument("--smoke-rollouts", type=int, default=3)
    ap.add_argument("--smoke-method", choices=tuple(methods.VARIANT_COMPONENTS), default="SA_EMA_RELEVANT")
    args = ap.parse_args()
    methods.install_hooks()
    selected = read_json(args.selection_file, {}) if args.selection_file else {}
    if args.selection_file and not args.selection_file.exists():
        raise FileNotFoundError(args.selection_file)
    cells = planned_cells(args.stage, selected)
    assert_design(cells, args.stage)
    if args.max_budget_steps <= 0 or args.max_budget_steps > CAP:
        raise ValueError("--max-budget-steps 必须在 1 到 500M 之间")
    if args.plan_only:
        print(json.dumps({"stage": args.stage, "cells": [asdict(c) for c in cells],
                          "planned_steps": sum(c.steps for c in cells),
                          "phase_caps": PHASE_CAPS, "total_cap": CAP},
                         ensure_ascii=False, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；不自动降级到 CPU")
    if args.smoke:
        # 多个完整 rollout 才能摊薄 Windows spawn 和模型初始化开销。
        if args.smoke_rollouts < 1 or args.smoke_rollouts > 10:
            raise ValueError("--smoke-rollouts 范围为 1..10")
        freeze = load_freeze(args.freeze, strict=False)
        if freeze:
            freeze["_sha256"] = digest(args.freeze)
            profile = profile_from_freeze(freeze)
        else:
            profile = v3.SpeedProfile("SMOKE_10x2048_b4096", 10, 2048, 4096)
        cell = Cell("SMOKE", args.smoke_method, "J4", 999, ROLLOUT * args.smoke_rollouts)
        started = time.perf_counter()
        result = run_cell(args.suite_root / "smoke", cell, freeze, profile,
                          args.device, retry_failed=True, smoke=True)
        write_json(args.suite_root / "speed_pilot.json", {
            "method": cell.method, "device": args.device,
            "profile": asdict(profile), "freeze_sha256": freeze.get("_sha256"),
            "train_sps": result["train_sps"], "steps": result["actual_steps"],
            "smoke_only": True,
        })
        print(f"SMOKE: {result}; wall={time.perf_counter()-started:.1f}s")
        return
    freeze = load_freeze(args.freeze, strict=True)
    freeze["_sha256"] = digest(args.freeze)
    final_test_lock = lock_final_test_bank(args.final_test_bank, freeze)
    profile = profile_from_freeze(freeze)
    root = args.suite_root.resolve()
    pilot = read_json(root / "speed_pilot.json", {})
    if not args.evaluate_only and (pilot.get("freeze_sha256") != freeze["_sha256"]
                                   or pilot.get("profile") != asdict(profile)
                                   or float(pilot.get("train_sps", 0)) < args.min_train_sps):
        raise RuntimeError(
            "正式训练先在同一冻结配置与 suite-root 上运行 --smoke，"
            f"并确认完整训练吞吐 >= {args.min_train_sps:.0f} SPS；"
            "当前 speed_pilot.json 缺失、过期或不达标"
        )
    root.mkdir(parents=True, exist_ok=True)
    suite_contract = {"freeze_sha256": freeze["_sha256"], "phase_caps": PHASE_CAPS,
                      "reward": "-N_active*delta_t", "test_bank_used": False,
                      "final_test_lock": final_test_lock}
    contract_path = root / "suite_contract.json"
    if contract_path.exists() and read_json(contract_path) != suite_contract:
        raise RuntimeError("suite 冻结合同变化；请另建 suite-root")
    write_json(contract_path, suite_contract)
    plan_path = root / args.stage / "stage_plan.json"
    plan = [asdict(c) for c in cells]
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError("stage selection 已变化，不能混入既有结果")
    write_json(plan_path, plan)
    if args.resume_cell:
        cells = [c for c in cells if c.key == args.resume_cell]
        if not cells:
            raise ValueError(f"该阶段不存在 --resume-cell={args.resume_cell}")
    if args.max_cells > 0:
        cells = cells[:args.max_cells]
    if args.evaluate_only:
        for c in cells:
            try:
                evaluate_cell(cell_dir(root, c), c, freeze)
            except Exception as exc:
                print(f"[EVAL FAILED] {c.key}: {exc!r}", flush=True)
                write_json(cell_dir(root, c) / "analysis" / "eval_error.json",
                           {"error": traceback.format_exc()})
        aggregate_stage(root, cells, args.stage)
        return
    errors: dict[str, int] = {}
    for c in cells:
        old = read_json(cell_dir(root, c) / "run_end.json", {})
        if old.get("status") == "COMPLETE" or (old.get("status") == "FAILED" and not args.retry_failed):
            print(f"[SKIP] {c.key} | {old.get('status')}", flush=True)
            continue
        spent, phases = all_spent(root)
        checkpoint = latest_checkpoint(cell_dir(root, c))
        resumable_step = checkpoint[0] if checkpoint is not None else 0
        # 失败后未落盘的步数已经计入历史开销，重跑时仍需重新预留预算。
        need = max(0, c.steps - resumable_step)
        if spent + need > args.max_budget_steps or phases[phase_of(c.stage)] + need > PHASE_CAPS[phase_of(c.stage)]:
            raise RuntimeError(f"预算闸门：已用 {spent:,}，本 cell 尚需 {need:,}")
        print(f"[TRAIN] {c.key} | {c.steps:,} steps | {profile.name} | {args.device}", flush=True)
        try:
            result = run_cell(root, c, freeze, profile, args.device,
                              retry_failed=args.retry_failed)
            print(f"[DONE] {c.key} | train_sps={result['train_sps']:.1f}", flush=True)
            if result["train_sps"] < args.min_train_sps:
                raise RuntimeError(f"训练吞吐 {result['train_sps']:.1f} < {args.min_train_sps:.1f}；先检查服务器负载")
        except Exception as exc:
            fingerprint = f"{type(exc).__name__}:{str(exc)[:160]}"
            errors[fingerprint] = errors.get(fingerprint, 0) + 1
            print(f"[FAILED] {c.key} | {fingerprint}", flush=True)
            if errors[fingerprint] >= 2 or "训练吞吐" in str(exc):
                raise RuntimeError("连续系统性失败或吞吐不达标；停止整批，保留已有 checkpoint") from exc
            # 单个独立 cell 失败时继续下一组，不自动掩盖错误。


if __name__ == "__main__":
    main()
