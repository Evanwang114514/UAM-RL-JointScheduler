# -*- coding: utf-8 -*-
"""
UAGMC Stage-0.5：Snapshot -> Committed-only -> Oracle 三层时间错位诊断（无训练）

目的
----
1. 沿用 Stage-0 的同一 UAGMC checkpoint、VecNormalize、passenger trace 与状态提取；
2. 对每个 passenger decision、每个 departure candidate，比较：
   - Snapshot：决策时刻当前状态；
   - Committed-only：只推进当前已经可见、已经承诺且有明确剩余 timer 的事件；
   - Oracle：checkpoint 真实后续 rollout 在 candidate realization horizon 的状态；
3. 回答：Stage-0 看到的 temporal drift 中，有多少至少可以由“合法 committed 信息”解释；
4. 不训练，不修改 checkpoint，不把未来未知 passenger 或未来 policy action 当作在线输入。

Committed-only 的严格边界
-------------------------
本脚本默认使用 strict timer-only projection：
- 当前已经 enroute 到某 departure vertiport 的 passenger：按 current_timer 推进；
- 当前已经 charging 的 eVTOL：按 remaining_charge_time 推进；
- 当前 waiting passenger 保留在 waiting 中；
- 当前 idle eVTOL 保留为 idle；
- 不生成未来未 reveal passenger；
- 不执行未来新的 passenger policy action；
- 不假装知道未来队列服务顺序、未来资源竞争或未来新航班。

因此它是“保守的 committed-only projection”，可能低估可恢复信息，但不会因为偷看 oracle 而高估。
如果后续要做 exact committed simulator fork，应单独实现并与本脚本结果对照。

依赖
----
请把本文件与 Stage-0 脚本放在 UAGMC 源码根目录：
    diagnose_uagmc_candidate_temporal_mismatch.py
    diagnose_uagmc_committed_vs_oracle.py

代码注释全部使用中文。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import diagnose_uagmc_candidate_temporal_mismatch as stage0
except Exception as exc:
    raise RuntimeError(
        "无法导入 Stage-0 脚本 diagnose_uagmc_candidate_temporal_mismatch.py。\n"
        "请确认两个 .py 文件都位于 UAGMC 源码根目录。"
    ) from exc


# ============================================================
# 基础工具
# ============================================================

SAFE_KEYS = (
    "waiting",
    "incoming",
    "charging_evtols",
    "idle_evtols",
    "avg_charge_time",
    "min_charge_time",
    "min_incoming_passenger_time",
    "avg_incoming_passenger_time",
)


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def percentile_or_nan(values: Sequence[float], q: float) -> float:
    vals = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.percentile(vals, q)) if vals else float("nan")


def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def argmin_stable(values: Dict[int, float]) -> int:
    return min(values.items(), key=lambda item: (item[1], item[0]))[0]


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    """不依赖 scipy 的平均秩实现，用于 Spearman 相关。"""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    if float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ============================================================
# 状态与 committed-only projection
# ============================================================


def evtol_state_name(evtol: Any) -> str:
    try:
        return str(evtol.state.name)
    except Exception:
        return str(getattr(evtol, "state", "UNKNOWN"))


def _finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def collect_committed_passenger_timers(
    scenario: Any,
    candidate_ids: Sequence[int],
) -> Dict[int, List[float]]:
    """读取当前已经 enroute 的 passenger；未来尚未 reveal 的 passenger 不进入。"""
    out: Dict[int, List[float]] = {int(v): [] for v in candidate_ids}
    persons_obj = getattr(scenario, "persons", None)
    persons = getattr(persons_obj, "persons", {}) if persons_obj is not None else {}

    for person in persons.values():
        if str(getattr(person, "state", "")).lower() != "enroute":
            continue
        try:
            vid = int(getattr(person, "origin_vertiport_id"))
        except Exception:
            continue
        if vid not in out:
            continue
        timer = _finite_float(getattr(person, "current_timer", None))
        if timer is not None:
            out[vid].append(max(0.0, timer))

    return out


def collect_local_evtols(scenario: Any, vid: int) -> List[Any]:
    """只读取当前已经位于候选站容器中的 eVTOL，不构造未来新飞机。"""
    try:
        return list(scenario.vertiports.evtols_at_vertiport.get(str(int(vid)), []))
    except Exception:
        return []


def project_committed_state(
    scenario: Any,
    vid: int,
    horizon: float,
    now_state: Dict[str, float],
    committed_passenger_timers: Dict[int, List[float]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    严格 timer-only committed projection。

    它只解析当前已经存在的剩余 timer，不模拟未来未知 passenger、未来 policy、
    未来队列服务顺序或新的资源竞争。
    """
    h = max(0.0, float(horizon))
    projected = dict(now_state)

    # 当前已经在 ground-access 的 passenger：到 horizon 前完成 access 的转入 waiting。
    passenger_timers = list(committed_passenger_timers.get(int(vid), []))
    arrived = [t for t in passenger_timers if t <= h + 1e-12]
    remaining = [max(0.0, t - h) for t in passenger_timers if t > h + 1e-12]

    projected["waiting"] = float(now_state.get("waiting", 0.0)) + float(len(arrived))
    projected["incoming"] = float(len(remaining))
    projected["min_incoming_passenger_time"] = (
        float(min(remaining)) if remaining else 0.0
    )
    projected["avg_incoming_passenger_time"] = (
        float(np.mean(remaining)) if remaining else 0.0
    )

    # 当前已经 charging 的 eVTOL：只推进明确的 remaining_charge_time。
    local_evtols = collect_local_evtols(scenario, int(vid))
    current_idle = 0
    current_charging = 0
    completed_charge = 0
    remaining_charge: List[float] = []
    unresolved_charging_without_timer = 0

    for ev in local_evtols:
        state_name = evtol_state_name(ev).upper()
        if state_name == "IDLE":
            current_idle += 1
            continue
        if state_name != "CHARGING":
            continue

        current_charging += 1
        timer = _finite_float(getattr(ev, "remaining_charge_time", None))
        if timer is None:
            unresolved_charging_without_timer += 1
            continue

        timer = max(0.0, timer)
        if timer <= h + 1e-12:
            completed_charge += 1
        else:
            remaining_charge.append(timer - h)

    projected["idle_evtols"] = float(current_idle + completed_charge)
    projected["charging_evtols"] = float(
        len(remaining_charge) + unresolved_charging_without_timer
    )
    projected["avg_charge_time"] = (
        float(np.mean(remaining_charge)) if remaining_charge else 0.0
    )
    projected["min_charge_time"] = (
        float(np.min(remaining_charge)) if remaining_charge else 0.0
    )

    # 这些量在 strict timer-only projection 中保持 snapshot 值，避免凭空模拟未知事件。
    for key in ("total_evtols", "total_capacity", "avg_flight_time"):
        projected[key] = float(now_state.get(key, 0.0))

    meta = {
        "committed_enroute_passengers_now": len(passenger_timers),
        "committed_passenger_arrivals_by_horizon": len(arrived),
        "committed_passengers_still_incoming": len(remaining),
        "local_evtols_now": len(local_evtols),
        "charging_evtols_with_timer": current_charging - unresolved_charging_without_timer,
        "charging_evtols_without_timer": unresolved_charging_without_timer,
        "charging_completions_by_horizon": completed_charge,
        "projection_mode": "strict_timer_only",
    }
    return projected, meta


def state_distance(
    a: Dict[str, float],
    b: Dict[str, float],
    keys: Sequence[str] = SAFE_KEYS,
) -> float:
    """与 Stage-0 一致的无权重相对漂移，但只在严格可投影量上计算。"""
    vals: List[float] = []
    for key in keys:
        av = float(a.get(key, 0.0))
        bv = float(b.get(key, 0.0))
        vals.append(abs(bv - av) / (1.0 + abs(av)))
    return float(np.mean(vals)) if vals else 0.0


def burden_proxy(state: Dict[str, float]) -> float:
    return float(state.get("waiting", 0.0)) + float(state.get("incoming", 0.0))


def pressure_proxy(state: Dict[str, float]) -> float:
    return (
        float(state.get("waiting", 0.0))
        + float(state.get("incoming", 0.0))
        - float(state.get("idle_evtols", 0.0))
    )


def component_alignment(
    snapshot: Dict[str, float],
    committed: Dict[str, float],
    oracle: Dict[str, float],
    keys: Sequence[str] = SAFE_KEYS,
) -> Tuple[int, int]:
    """统计 committed delta 与 oracle delta 的方向是否一致。"""
    aligned = 0
    eligible = 0
    for key in keys:
        s = float(snapshot.get(key, 0.0))
        c = float(committed.get(key, 0.0))
        o = float(oracle.get(key, 0.0))
        do = o - s
        dc = c - s
        if abs(do) <= 1e-12:
            continue
        eligible += 1
        if abs(dc) <= 1e-12:
            continue
        if math.copysign(1.0, dc) == math.copysign(1.0, do):
            aligned += 1
    return aligned, eligible


# ============================================================
# 命令行
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="UAGMC Stage-0.5 Snapshot/Committed/Oracle 无训练诊断"
    )
    p.add_argument("--project-root", default=str(Path.cwd()))
    p.add_argument("--uagmc-root", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--passengers", default=None)
    p.add_argument("--candidates", default=None)
    p.add_argument("--to-vertiport", type=int, default=2)
    p.add_argument("--max-time", type=int, default=600)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument(
        "--output-dir",
        default="diagnostics/uagmc_committed_vs_oracle",
    )
    return p.parse_args()


# ============================================================
# 主流程
# ============================================================


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(project_root)

    uagmc_root = stage0.discover_uagmc_root(project_root, args.uagmc_root)
    model_path = stage0.discover_model(uagmc_root, project_root, args.model)
    vecnorm_path = stage0.discover_vecnorm(
        uagmc_root, project_root, model_path, args.vecnorm
    )
    passenger_path = stage0.discover_passenger_file(
        uagmc_root, project_root, args.passengers
    )

    sys.path.insert(0, str(uagmc_root))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from utilss.make_env import make_env

    initial_candidates = (
        [int(x.strip()) for x in args.candidates.split(",") if x.strip()]
        if args.candidates
        else [0, 1]
    )

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = project_root / args.output_dir / run_stamp
    monitor_dir = output_root / "_monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv(
        [
            make_env(
                max_time=args.max_time,
                log_dir=monitor_dir,
                env_index=0,
                candidate_from_vertiports=initial_candidates,
                to_vertiport=args.to_vertiport,
                person_spawn_file=str(passenger_path),
                enable_logger=False,
            )
        ]
    )

    if vecnorm_path is not None:
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    device = stage0.resolve_device(args.device)
    model = PPO.load(str(model_path), env=env, device=device)

    wrapper = stage0.unwrap_uagmc_wrapper(env)
    scenario = wrapper.env
    candidate_ids = stage0.get_candidate_ids(wrapper, args.candidates)

    if len(candidate_ids) != int(wrapper.action_space.n):
        raise RuntimeError(
            f"candidate 数量 {len(candidate_ids)} 与 action_space.n={wrapper.action_space.n} 不一致。"
        )

    print("=" * 112)
    print("UAGMC STAGE-0.5 | SNAPSHOT -> COMMITTED-ONLY -> ORACLE | NO TRAINING")
    print("=" * 112)
    print(f"Project root     : {project_root}")
    print(f"UAGMC root       : {uagmc_root}")
    print(f"Model            : {model_path}")
    print(f"VecNormalize     : {vecnorm_path}")
    print(f"Passengers       : {passenger_path}")
    print(f"Candidates       : {candidate_ids}")
    print(f"Projection mode  : strict_timer_only")
    print(f"Output           : {output_root}")
    print("-" * 112)

    # 先沿 checkpoint 的真实 rollout 记录每个物理时刻的 oracle 状态。
    snapshots: Dict[int, Dict[int, Dict[str, float]]] = {}
    decisions: List[Dict[str, Any]] = []

    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        sim_time = int(getattr(scenario, "time", step_count))

        if sim_time not in snapshots:
            snapshots[sim_time] = stage0.extract_vertiport_state(
                scenario, candidate_ids
            )

        pid = stage0.get_waiting_pid(wrapper)
        probs = stage0.get_policy_probs(model, obs)
        action, _ = model.predict(obs, deterministic=True)
        action_index = int(np.asarray(action).reshape(-1)[0])

        if pid is not None:
            access_times = stage0.estimate_access_times(
                scenario, pid, candidate_ids
            )
            now_state = stage0.extract_vertiport_state(
                scenario, candidate_ids
            )
            passenger_timers = collect_committed_passenger_timers(
                scenario, candidate_ids
            )

            if probs is None or len(probs) != len(candidate_ids):
                policy_probs = [float("nan")] * len(candidate_ids)
                policy_margin = float("nan")
            else:
                policy_probs = [float(x) for x in probs]
                ordered = sorted(policy_probs, reverse=True)
                policy_margin = (
                    float(ordered[0] - ordered[1])
                    if len(ordered) >= 2
                    else 1.0
                )

            committed_by_candidate: Dict[int, Dict[str, float]] = {}
            meta_by_candidate: Dict[int, Dict[str, Any]] = {}
            for vid in candidate_ids:
                projected, meta = project_committed_state(
                    scenario=scenario,
                    vid=int(vid),
                    horizon=float(access_times[vid]),
                    now_state=now_state[vid],
                    committed_passenger_timers=passenger_timers,
                )
                committed_by_candidate[int(vid)] = projected
                meta_by_candidate[int(vid)] = meta

            decisions.append(
                {
                    "decision_id": len(decisions),
                    "sim_time": sim_time,
                    "pid": pid,
                    "chosen_action_index": action_index,
                    "chosen_vertiport": int(candidate_ids[action_index]),
                    "access_times": access_times,
                    "policy_probs": policy_probs,
                    "policy_margin": policy_margin,
                    "snapshot": now_state,
                    "committed": committed_by_candidate,
                    "committed_meta": meta_by_candidate,
                }
            )

        obs, _, dones, _ = env.step(action)
        done = bool(np.asarray(dones).reshape(-1)[0])
        step_count += 1

        if step_count > args.max_time + 1000:
            print("[警告] 超过安全 step 上限，强制结束。")
            break

    final_time = int(getattr(scenario, "time", step_count))
    if final_time not in snapshots:
        snapshots[final_time] = stage0.extract_vertiport_state(
            scenario, candidate_ids
        )

    available_times = sorted(snapshots.keys())

    def future_snapshot_time(target_time: float) -> Optional[int]:
        target = int(math.ceil(target_time))
        for t in available_times:
            if t >= target:
                return t
        return None

    # 汇总三层状态距离与 ranking 变化。
    candidate_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    d_sc_all: List[float] = []
    d_so_all: List[float] = []
    d_co_all: List[float] = []
    error_gain_all: List[float] = []
    raw_ratio_all: List[float] = []
    positive_ratio_all: List[float] = []
    alignment_rates: List[float] = []
    access_all: List[float] = []

    for decision in decisions:
        snapshot_waiting: Dict[int, float] = {}
        committed_waiting: Dict[int, float] = {}
        oracle_waiting: Dict[int, float] = {}

        snapshot_burden: Dict[int, float] = {}
        committed_burden: Dict[int, float] = {}
        oracle_burden: Dict[int, float] = {}

        snapshot_pressure: Dict[int, float] = {}
        committed_pressure: Dict[int, float] = {}
        oracle_pressure: Dict[int, float] = {}

        per_dec_d_sc: List[float] = []
        per_dec_d_so: List[float] = []
        per_dec_d_co: List[float] = []
        per_dec_gain: List[float] = []
        per_dec_ratio: List[float] = []
        per_dec_align_num = 0
        per_dec_align_den = 0
        all_future_available = True

        for action_index, vid in enumerate(candidate_ids):
            vid = int(vid)
            h = float(decision["access_times"][vid])
            target_time = float(decision["sim_time"]) + h
            future_t = future_snapshot_time(target_time)

            s = decision["snapshot"][vid]
            c = decision["committed"][vid]
            meta = decision["committed_meta"][vid]

            row: Dict[str, Any] = {
                "decision_id": decision["decision_id"],
                "sim_time": decision["sim_time"],
                "pid": decision["pid"],
                "candidate_action_index": action_index,
                "candidate_vertiport": vid,
                "checkpoint_choice": int(vid == decision["chosen_vertiport"]),
                "access_time": h,
                "candidate_realization_time": target_time,
                "future_snapshot_time": future_t if future_t is not None else "",
                "policy_prob": decision["policy_probs"][action_index],
                "policy_margin": decision["policy_margin"],
                "projection_mode": meta["projection_mode"],
            }

            for key, value in meta.items():
                if key != "projection_mode":
                    row[key] = value

            for key in SAFE_KEYS:
                row[f"snapshot_{key}"] = float(s.get(key, 0.0))
                row[f"committed_{key}"] = float(c.get(key, 0.0))
                row[f"delta_committed_{key}"] = (
                    float(c.get(key, 0.0)) - float(s.get(key, 0.0))
                )

            snapshot_waiting[vid] = float(s.get("waiting", 0.0))
            committed_waiting[vid] = float(c.get("waiting", 0.0))
            snapshot_burden[vid] = burden_proxy(s)
            committed_burden[vid] = burden_proxy(c)
            snapshot_pressure[vid] = pressure_proxy(s)
            committed_pressure[vid] = pressure_proxy(c)

            d_sc = state_distance(s, c)
            row["drift_snapshot_to_committed"] = d_sc
            per_dec_d_sc.append(d_sc)
            d_sc_all.append(d_sc)
            access_all.append(h)

            if future_t is None:
                all_future_available = False
                row["future_available"] = 0
                candidate_rows.append(row)
                continue

            o = snapshots[future_t][vid]
            row["future_available"] = 1
            for key in SAFE_KEYS:
                row[f"oracle_{key}"] = float(o.get(key, 0.0))
                row[f"delta_oracle_{key}"] = (
                    float(o.get(key, 0.0)) - float(s.get(key, 0.0))
                )
                row[f"residual_committed_to_oracle_{key}"] = (
                    float(o.get(key, 0.0)) - float(c.get(key, 0.0))
                )

            d_so = state_distance(s, o)
            d_co = state_distance(c, o)
            gain = d_so - d_co
            raw_ratio = gain / d_so if d_so > 1e-12 else float("nan")
            positive_ratio = (
                min(1.0, max(0.0, raw_ratio))
                if np.isfinite(raw_ratio)
                else float("nan")
            )

            aligned, eligible = component_alignment(s, c, o)
            align_rate = aligned / eligible if eligible > 0 else float("nan")

            row["drift_snapshot_to_oracle"] = d_so
            row["drift_committed_to_oracle"] = d_co
            row["oracle_error_reduction"] = gain
            row["oracle_error_reduction_ratio_raw"] = raw_ratio
            row["oracle_error_reduction_ratio_clipped"] = positive_ratio
            row["component_direction_alignment_rate"] = align_rate

            per_dec_d_so.append(d_so)
            per_dec_d_co.append(d_co)
            per_dec_gain.append(gain)
            if np.isfinite(raw_ratio):
                per_dec_ratio.append(raw_ratio)
            per_dec_align_num += aligned
            per_dec_align_den += eligible

            d_so_all.append(d_so)
            d_co_all.append(d_co)
            error_gain_all.append(gain)
            if np.isfinite(raw_ratio):
                raw_ratio_all.append(raw_ratio)
                positive_ratio_all.append(positive_ratio)
            if np.isfinite(align_rate):
                alignment_rates.append(align_rate)

            oracle_waiting[vid] = float(o.get("waiting", 0.0))
            oracle_burden[vid] = burden_proxy(o)
            oracle_pressure[vid] = pressure_proxy(o)

            candidate_rows.append(row)

        if not all_future_available or len(oracle_waiting) != len(candidate_ids):
            continue

        sw = argmin_stable(snapshot_waiting)
        cw = argmin_stable(committed_waiting)
        ow = argmin_stable(oracle_waiting)

        sb = argmin_stable(snapshot_burden)
        cb = argmin_stable(committed_burden)
        ob = argmin_stable(oracle_burden)

        sp = argmin_stable(snapshot_pressure)
        cp = argmin_stable(committed_pressure)
        op = argmin_stable(oracle_pressure)

        access_values = [float(x) for x in decision["access_times"].values()]

        decision_rows.append(
            {
                "decision_id": decision["decision_id"],
                "sim_time": decision["sim_time"],
                "pid": decision["pid"],
                "chosen_vertiport": decision["chosen_vertiport"],
                "access_delay_min": min(access_values),
                "access_delay_max": max(access_values),
                "access_delay_spread": max(access_values) - min(access_values),
                "policy_margin": decision["policy_margin"],
                "mean_drift_snapshot_to_committed": mean_or_nan(per_dec_d_sc),
                "mean_drift_snapshot_to_oracle": mean_or_nan(per_dec_d_so),
                "mean_drift_committed_to_oracle": mean_or_nan(per_dec_d_co),
                "mean_oracle_error_reduction": mean_or_nan(per_dec_gain),
                "mean_oracle_error_reduction_ratio_raw": mean_or_nan(per_dec_ratio),
                "component_direction_alignment_rate": (
                    per_dec_align_num / per_dec_align_den
                    if per_dec_align_den > 0
                    else float("nan")
                ),
                "snapshot_best_waiting": sw,
                "committed_best_waiting": cw,
                "oracle_best_waiting": ow,
                "flip_snapshot_to_committed_waiting": int(sw != cw),
                "flip_snapshot_to_oracle_waiting": int(sw != ow),
                "flip_committed_to_oracle_waiting": int(cw != ow),
                "snapshot_best_burden": sb,
                "committed_best_burden": cb,
                "oracle_best_burden": ob,
                "flip_snapshot_to_committed_burden": int(sb != cb),
                "flip_snapshot_to_oracle_burden": int(sb != ob),
                "flip_committed_to_oracle_burden": int(cb != ob),
                "snapshot_best_pressure": sp,
                "committed_best_pressure": cp,
                "oracle_best_pressure": op,
                "flip_snapshot_to_committed_pressure": int(sp != cp),
                "flip_snapshot_to_oracle_pressure": int(sp != op),
                "flip_committed_to_oracle_pressure": int(cp != op),
            }
        )

    # 用 absolute candidate horizon 分桶，而不是继续把 delay spread 当主假设。
    horizon_bin_rows: List[Dict[str, Any]] = []
    usable_candidates = [
        r for r in candidate_rows
        if int(r.get("future_available", 0)) == 1
        and np.isfinite(float(r.get("drift_snapshot_to_oracle", float("nan"))))
    ]

    if usable_candidates:
        horizons = np.asarray([float(r["access_time"]) for r in usable_candidates])
        edges = np.unique(np.quantile(horizons, [0.0, 0.25, 0.50, 0.75, 1.0]))
        for i in range(len(edges) - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if i == len(edges) - 2:
                rows = [r for r in usable_candidates if lo <= float(r["access_time"]) <= hi]
            else:
                rows = [r for r in usable_candidates if lo <= float(r["access_time"]) < hi]
            if not rows:
                continue
            horizon_bin_rows.append(
                {
                    "horizon_lo": lo,
                    "horizon_hi": hi,
                    "n": len(rows),
                    "mean_snapshot_to_committed": mean_or_nan(
                        [r["drift_snapshot_to_committed"] for r in rows]
                    ),
                    "mean_snapshot_to_oracle": mean_or_nan(
                        [r["drift_snapshot_to_oracle"] for r in rows]
                    ),
                    "mean_committed_to_oracle": mean_or_nan(
                        [r["drift_committed_to_oracle"] for r in rows]
                    ),
                    "mean_oracle_error_reduction": mean_or_nan(
                        [r["oracle_error_reduction"] for r in rows]
                    ),
                    "mean_oracle_error_reduction_ratio_raw": mean_or_nan(
                        [r["oracle_error_reduction_ratio_raw"] for r in rows]
                    ),
                    "positive_improvement_rate": mean_or_nan(
                        [int(float(r["oracle_error_reduction"]) > 0.0) for r in rows]
                    ),
                }
            )

    # 输出摘要。
    positive_improvement_rate = mean_or_nan(
        [int(x > 0.0) for x in error_gain_all]
    )

    summary = {
        "stage": "Stage-0.5 snapshot-committed-oracle no-training diagnostic",
        "projection_mode": "strict_timer_only",
        "project_root": str(project_root),
        "uagmc_root": str(uagmc_root),
        "model": str(model_path),
        "vecnormalize": str(vecnorm_path) if vecnorm_path else None,
        "passenger_trace": str(passenger_path),
        "candidate_vertiports": candidate_ids,
        "recorded_decisions": len(decisions),
        "analyzable_decisions": len(decision_rows),
        "candidate_rows": len(candidate_rows),
        "safe_state_keys": list(SAFE_KEYS),
        "drift": {
            "mean_snapshot_to_committed": mean_or_nan(d_sc_all),
            "mean_snapshot_to_oracle": mean_or_nan(d_so_all),
            "mean_committed_to_oracle": mean_or_nan(d_co_all),
            "median_snapshot_to_oracle": percentile_or_nan(d_so_all, 50),
            "p90_snapshot_to_oracle": percentile_or_nan(d_so_all, 90),
        },
        "committed_explanatory_power": {
            "mean_oracle_error_reduction": mean_or_nan(error_gain_all),
            "median_oracle_error_reduction": percentile_or_nan(error_gain_all, 50),
            "positive_improvement_rate": positive_improvement_rate,
            "mean_error_reduction_ratio_raw": mean_or_nan(raw_ratio_all),
            "median_error_reduction_ratio_raw": percentile_or_nan(raw_ratio_all, 50),
            "mean_error_reduction_ratio_clipped": mean_or_nan(positive_ratio_all),
            "mean_component_direction_alignment_rate": mean_or_nan(alignment_rates),
        },
        "horizon_relationship": {
            "rho_access_vs_snapshot_to_committed": spearman_rho(access_all, d_sc_all),
            "rho_access_vs_snapshot_to_oracle": spearman_rho(
                [float(r["access_time"]) for r in usable_candidates],
                [float(r["drift_snapshot_to_oracle"]) for r in usable_candidates],
            ) if usable_candidates else float("nan"),
            "rho_access_vs_committed_to_oracle": spearman_rho(
                [float(r["access_time"]) for r in usable_candidates],
                [float(r["drift_committed_to_oracle"]) for r in usable_candidates],
            ) if usable_candidates else float("nan"),
        },
        "ranking_flip_rates": {
            "snapshot_to_committed_waiting": mean_or_nan(
                [r["flip_snapshot_to_committed_waiting"] for r in decision_rows]
            ),
            "snapshot_to_oracle_waiting": mean_or_nan(
                [r["flip_snapshot_to_oracle_waiting"] for r in decision_rows]
            ),
            "committed_to_oracle_waiting": mean_or_nan(
                [r["flip_committed_to_oracle_waiting"] for r in decision_rows]
            ),
            "snapshot_to_committed_burden": mean_or_nan(
                [r["flip_snapshot_to_committed_burden"] for r in decision_rows]
            ),
            "snapshot_to_oracle_burden": mean_or_nan(
                [r["flip_snapshot_to_oracle_burden"] for r in decision_rows]
            ),
            "committed_to_oracle_burden": mean_or_nan(
                [r["flip_committed_to_oracle_burden"] for r in decision_rows]
            ),
            "snapshot_to_committed_pressure": mean_or_nan(
                [r["flip_snapshot_to_committed_pressure"] for r in decision_rows]
            ),
            "snapshot_to_oracle_pressure": mean_or_nan(
                [r["flip_snapshot_to_oracle_pressure"] for r in decision_rows]
            ),
            "committed_to_oracle_pressure": mean_or_nan(
                [r["flip_committed_to_oracle_pressure"] for r in decision_rows]
            ),
        },
        "horizon_bins": horizon_bin_rows,
        "interpretation_guardrails": [
            "Committed-only 是 strict timer-only projection，不是完整 world model。",
            "它只推进当前已知 enroute passenger 与 charging eVTOL 的明确剩余 timer。",
            "它不读取未来未 reveal passenger，也不执行未来新的 passenger policy action。",
            "由于未模拟未来队列服务顺序和资源竞争，它可能低估 committed information 的真实可解释比例。",
            "Oracle 来自 checkpoint 真实后续 rollout，只用于离线诊断，不能作为在线策略输入。",
            "error_reduction_ratio_raw 可以为负；负值表示该保守 projection 在该样本上没有把状态拉近 oracle。",
            "本实验仍不证明 candidate-relative representation 会提升 ATT；它只检查合法 committed 信息是否解释了 temporal mismatch。",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "candidate_detail.csv", candidate_rows)
    write_csv(output_root / "decision_summary.csv", decision_rows)
    write_csv(output_root / "access_horizon_bins.csv", horizon_bin_rows)

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(summary), f, ensure_ascii=False, indent=2)

    with (output_root / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            jsonable(
                {
                    "project_root": project_root,
                    "uagmc_root": uagmc_root,
                    "model": model_path,
                    "vecnormalize": vecnorm_path,
                    "passenger_trace": passenger_path,
                    "candidate_vertiports": candidate_ids,
                    "to_vertiport": args.to_vertiport,
                    "device": device,
                    "projection_mode": "strict_timer_only",
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    env.close()

    print("\n" + "=" * 112)
    print("STAGE-0.5 SUMMARY")
    print("=" * 112)
    print(f"Recorded decisions                 : {len(decisions)}")
    print(f"Analyzable decisions               : {len(decision_rows)}")
    print(f"Mean Snapshot -> Committed drift   : {summary['drift']['mean_snapshot_to_committed']:.6f}")
    print(f"Mean Snapshot -> Oracle drift      : {summary['drift']['mean_snapshot_to_oracle']:.6f}")
    print(f"Mean Committed -> Oracle residual  : {summary['drift']['mean_committed_to_oracle']:.6f}")
    print(f"Mean oracle error reduction        : {summary['committed_explanatory_power']['mean_oracle_error_reduction']:.6f}")
    print(f"Positive improvement rate          : {summary['committed_explanatory_power']['positive_improvement_rate']:.4f}")
    print(f"Mean raw explanation ratio         : {summary['committed_explanatory_power']['mean_error_reduction_ratio_raw']:.4f}")
    print(f"Direction alignment rate           : {summary['committed_explanatory_power']['mean_component_direction_alignment_rate']:.4f}")
    print(f"rho(access, Snapshot->Committed)    : {summary['horizon_relationship']['rho_access_vs_snapshot_to_committed']:.4f}")
    print(f"rho(access, Snapshot->Oracle)       : {summary['horizon_relationship']['rho_access_vs_snapshot_to_oracle']:.4f}")
    print("-" * 112)
    print("注意：Committed-only 是严格 timer-only 保守投影；Oracle 仅用于离线诊断。")
    print(f"结果目录：{output_root}")
    print("=" * 112)


if __name__ == "__main__":
    main()
