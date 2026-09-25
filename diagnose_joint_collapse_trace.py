# -*- coding: utf-8 -*-
"""
Detailed deterministic rollout diagnostic for V4 Joint UAM collapse.

READ-ONLY with respect to training results:
- loads an existing checkpoint + VecNormalize
- runs one deterministic evaluation episode
- writes diagnostic CSV/JSON/ZIP only under <run-root>/diagnostics/

Main outputs
------------
decision_trace.csv
    One row per simulator step/decision:
    - four joint-action probabilities
    - chosen passenger/aircraft targets
    - focal passenger/access-time/effect-time projections
    - exact V0/V1 queue lengths
    - aircraft counts/states at V0/V1/V2
    - hub eligible aircraft before action
    - dispatch success / no-eligible / rejected-with-eligible
    - generated/finished/backlog before and after the step

window_summary.csv
    Rolling non-overlapping windows (default 50 steps) summarizing:
    action shares, dispatch starvation, queue/backlog growth, throughput.

aircraft_snapshots.csv
    Per-aircraft state every --snapshot-every steps plus the terminal state.

summary.json
    Formal episode metrics + action/dispatch statistics + candidate collapse onset.

UPLOAD_THIS_*.zip
    Small bundle containing the diagnostic outputs for upload/analysis.

Examples
--------
python diagnose_joint_collapse_trace.py ^
  --run-root "serial_runs\\uam_jointfirst60m_v4_seed1_20260924_010334" ^
  --cell J0__CURRENT --step 600000 --eval-seed 123

python diagnose_joint_collapse_trace.py ^
  --run-root "serial_runs\\uam_jointfirst60m_v4_seed1_20260924_010334" ^
  --cell J0__R_EVENTQ --step 600000 --eval-seed 123
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO

import train_uam_60m_jointfirst_v4_deferred_joint_eval as v4

v3 = v4.v3


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(v, ensure_ascii=False)
                rr[k] = v
            w.writerow(rr)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def parse_cell(cell: str) -> Tuple[str, str]:
    s = str(cell).strip()
    if "__" not in s:
        raise ValueError("--cell must look like J0__CURRENT")
    env_key, method = s.split("__", 1)
    env_key = env_key.upper()
    if env_key not in v3.ENV_SPECS or not v3.ENV_SPECS[env_key].joint:
        raise ValueError(f"{env_key} is not a Joint environment key")
    return env_key, method


def get_person_dict(scenario: Any) -> Dict[str, Any]:
    obj = getattr(scenario, "persons", None)
    raw = getattr(obj, "persons", {}) if obj is not None else {}
    return {str(k): v for k, v in (raw or {}).items()}


def finished_set(scenario: Any) -> set[str]:
    return {str(x) for x in (getattr(scenario, "finished_ids", []) or [])}


def queue_lengths(scenario: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    vp_builder = getattr(scenario, "vertiports", None)
    vp_list = getattr(vp_builder, "vertiport_list", {}) if vp_builder is not None else {}
    for vid, vp in (vp_list or {}).items():
        out[str(vid)] = len(list(getattr(vp, "person_list", []) or []))
    return out


def all_evtols(scenario: Any) -> List[Any]:
    raw = getattr(scenario, "_all_evtols", None)
    if isinstance(raw, dict) and raw:
        return list(raw.values())

    seen: Dict[str, Any] = {}
    vp_builder = getattr(scenario, "vertiports", None)
    at_vp = getattr(vp_builder, "evtols_at_vertiport", {}) if vp_builder is not None else {}
    for xs in (at_vp or {}).values():
        for e in xs or []:
            seen[str(getattr(e, "id", id(e)))] = e

    evtol_builder = getattr(scenario, "evtols", None)
    for attr in ("evtols", "_all_evtols"):
        d = getattr(evtol_builder, attr, None) if evtol_builder is not None else None
        if isinstance(d, dict):
            for e in d.values():
                seen[str(getattr(e, "id", id(e)))] = e
    return list(seen.values())


def state_name(e: Any) -> str:
    try:
        return str(v3.mx.state_name(e))
    except Exception:
        x = getattr(e, "state", "UNKNOWN")
        return str(getattr(x, "name", x))


def passenger_ids(e: Any) -> List[str]:
    try:
        return [str(x) for x in v3.mx.passenger_ids(e)]
    except Exception:
        return [str(x) for x in (getattr(e, "passenger_ids", []) or [])]


def aircraft_aggregate(scenario: Any) -> Dict[str, Any]:
    evs = all_evtols(scenario)
    state_counts = Counter()
    current_counts = Counter()
    idle_counts = Counter()
    charging_counts = Counter()
    target_counts = Counter()
    in_flight_target_counts = Counter()

    for e in evs:
        st = state_name(e)
        cur = str(getattr(e, "current_vertiport_id", ""))
        tgt = str(getattr(e, "target_vertiport_id", ""))
        state_counts[st] += 1
        if cur:
            current_counts[cur] += 1
        if tgt:
            target_counts[tgt] += 1
        if st == "IDLE" and cur:
            idle_counts[cur] += 1
        if st == "CHARGING" and cur:
            charging_counts[cur] += 1
        if st not in ("IDLE", "CHARGING") and tgt:
            in_flight_target_counts[tgt] += 1

    hub = str(v3.DESTINATION)
    local = list(
        getattr(getattr(scenario, "vertiports", None), "evtols_at_vertiport", {}).get(hub, [])
        or []
    )
    eligible = [
        e for e in local
        if (
            state_name(e) == "IDLE"
            and not passenger_ids(e)
            and not v3.base.is_turnaround_busy(e)
        )
    ]

    out: Dict[str, Any] = {
        "aircraft_total": len(evs),
        "hub_eligible": len(eligible),
        "aircraft_state_counts_json": dict(state_counts),
    }
    for vid in ("0", "1", str(v3.DESTINATION)):
        out[f"aircraft_at_V{vid}"] = int(current_counts.get(vid, 0))
        out[f"idle_at_V{vid}"] = int(idle_counts.get(vid, 0))
        out[f"charging_at_V{vid}"] = int(charging_counts.get(vid, 0))
        out[f"targeting_V{vid}"] = int(target_counts.get(vid, 0))
        out[f"inflight_target_V{vid}"] = int(in_flight_target_counts.get(vid, 0))
    return out


def aircraft_snapshot_rows(scenario: Any, time_value: int) -> List[Dict[str, Any]]:
    rows = []
    for e in sorted(all_evtols(scenario), key=lambda x: str(getattr(x, "id", ""))):
        spec = getattr(e, "spec", None)
        rows.append({
            "time": int(time_value),
            "aircraft_id": str(getattr(e, "id", "")),
            "state": state_name(e),
            "current_vertiport_id": getattr(e, "current_vertiport_id", None),
            "target_vertiport_id": getattr(e, "target_vertiport_id", None),
            "passenger_ids": passenger_ids(e),
            "remaining_flight_time": getattr(e, "remaining_flight_time", None),
            "battery_kwh": getattr(e, "battery_kwh", None),
            "battery_capacity_kwh": getattr(spec, "battery_capacity_kwh", None),
            "turnaround_busy": bool(getattr(e, "_e345_turnaround_busy", False)),
            "turnaround_release_time": getattr(e, "_e345_turnaround_release_time", None),
            "charge_queue_enter_time": getattr(e, "_e6_charge_queue_enter_time", None),
        })
    return rows


def focal_pid(env: Any) -> str:
    try:
        uam = v3.mx.find_uam_wrapper(env)
        waiting = list((getattr(uam, "state", {}) or {}).get("waiting_decisions", []) or [])
        return str(waiting[0]) if waiting else ""
    except Exception:
        return ""


def project_candidate(
    scenario: Any,
    spec: Any,
    person: Optional[Any],
    vid: int,
) -> Dict[str, float]:
    try:
        tau = float(v3._access_time(scenario, person, int(vid))) if person is not None else 0.0
    except Exception:
        tau = 0.0
    try:
        p0 = list(v3._project(scenario, spec, int(vid), 0.0))
    except Exception:
        p0 = [float("nan")] * 10
    try:
        pe = list(v3._project(scenario, spec, int(vid), tau))
    except Exception:
        pe = [float("nan")] * 10

    # These indices are exactly the ones consumed by the current V3.1 TDM card:
    # [1]=projected queue, [2]=committed incoming, [5]=projected supply.
    def at(xs: Sequence[float], i: int) -> float:
        return fnum(xs[i]) if i < len(xs) else float("nan")

    return {
        f"tau_access_V{vid}": tau,
        f"proj_now_queue_V{vid}": at(p0, 1),
        f"proj_now_committed_incoming_V{vid}": at(p0, 2),
        f"proj_now_supply_V{vid}": at(p0, 5),
        f"proj_effect_queue_V{vid}": at(pe, 1),
        f"proj_effect_committed_incoming_V{vid}": at(pe, 2),
        f"proj_effect_supply_V{vid}": at(pe, 5),
    }


def resolve_model_paths(run_dir: Path, step: int) -> Tuple[Path, Path, str]:
    mp = run_dir / "checkpoints" / f"uam_ppo_{step}_steps.zip"
    vp = run_dir / "checkpoints" / f"uam_ppo_vecnormalize_{step}_steps.pkl"
    if mp.exists() and vp.exists():
        return mp, vp, "checkpoint"

    if int(step) == 600000:
        mp2 = run_dir / "final_rl_model.zip"
        vp2 = run_dir / "final_vec_normalize.pkl"
        if mp2.exists() and vp2.exists():
            return mp2, vp2, "final_fallback"

    raise FileNotFoundError(
        f"Could not find model/VecNormalize for step={step} under {run_dir}"
    )


def summarize_windows(
    rows: Sequence[Dict[str, Any]],
    window: int,
    collapse_no_eligible_threshold: float,
    collapse_backlog_threshold: int,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    rows = list(rows)
    out: List[Dict[str, Any]] = []
    first_collapse: Optional[Dict[str, Any]] = None

    for i in range(0, len(rows), max(1, int(window))):
        xs = rows[i:i + max(1, int(window))]
        if not xs:
            continue

        n = len(xs)
        pair = Counter(int(r["joint_action"]) for r in xs)
        p = Counter(int(r["passenger_action"]) for r in xs)
        a = Counter(int(r["aircraft_action"]) for r in xs)
        statuses = Counter(str(r["dispatch_status"]) for r in xs)

        q0 = [fnum(r.get("queue_V0_after"), 0.0) for r in xs]
        q1 = [fnum(r.get("queue_V1_after"), 0.0) for r in xs]
        backlog = [fnum(r.get("backlog_after"), 0.0) for r in xs]
        eligible = [fnum(r.get("hub_eligible_before"), 0.0) for r in xs]

        rec = {
            "window_index": len(out),
            "row_start": i,
            "row_end": i + n - 1,
            "time_start": xs[0]["time_before"],
            "time_end": xs[-1]["time_after"],
            "n_steps": n,
            "pair_00_share": pair[0] / n,
            "pair_01_share": pair[1] / n,
            "pair_10_share": pair[2] / n,
            "pair_11_share": pair[3] / n,
            "passenger_V0_share": p[0] / n,
            "passenger_V1_share": p[1] / n,
            "aircraft_V0_share": a[0] / n,
            "aircraft_V1_share": a[1] / n,
            "dispatch_success_rate": statuses["SUCCESS"] / n,
            "dispatch_no_eligible_rate": statuses["NO_ELIGIBLE_HUB"] / n,
            "dispatch_rejected_with_eligible_rate": statuses["REJECTED_WITH_ELIGIBLE"] / n,
            "hub_eligible_mean": float(np.mean(eligible)),
            "queue_V0_mean": float(np.mean(q0)),
            "queue_V1_mean": float(np.mean(q1)),
            "queue_V0_end": q0[-1],
            "queue_V1_end": q1[-1],
            "backlog_mean": float(np.mean(backlog)),
            "backlog_end": backlog[-1],
            "generated_end": xs[-1]["generated_after"],
            "finished_end": xs[-1]["finished_after"],
            "finished_delta": int(xs[-1]["finished_after"]) - int(xs[0]["finished_before"]),
        }
        out.append(rec)

        if (
            first_collapse is None
            and rec["dispatch_no_eligible_rate"] >= float(collapse_no_eligible_threshold)
            and rec["backlog_end"] >= int(collapse_backlog_threshold)
        ):
            first_collapse = dict(rec)

    return out, first_collapse


def zip_outputs(out_dir: Path) -> Path:
    zpath = out_dir / f"UPLOAD_THIS_{out_dir.name}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.iterdir()):
            if p == zpath or not p.is_file():
                continue
            zf.write(p, arcname=p.name)
    return zpath


def run(args: argparse.Namespace) -> int:
    v3.legacy.assert_p0_patch()
    v4.install_v4_hooks()

    env_key, method_id = parse_cell(args.cell)
    spec = v3.ENV_SPECS[env_key]

    root = Path(args.run_root).expanduser().resolve()
    run_dir = root / args.cell
    if not run_dir.is_dir():
        # Windows is case-insensitive, but keep a small robust search for copied archives.
        matches = [p for p in root.iterdir() if p.is_dir() and p.name.upper() == args.cell.upper()]
        if not matches:
            raise FileNotFoundError(run_dir)
        run_dir = matches[0]

    model_path, vec_path, source_kind = resolve_model_paths(run_dir, int(args.step))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = (
            root / "diagnostics"
            / f"joint_trace_{args.cell}_{int(args.step)//1000}k_seed{args.eval_seed}_{stamp}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    v3.seed_all(int(args.eval_seed))
    raw = v3.build_eval_env(
        env_key=env_key,
        method_id=method_id,
        run_dir=out_dir / "_eval_monitor",
        max_time=int(args.max_time),
    )
    env = v3.TauPreservingVecNormalize.load(str(vec_path), raw)
    env.training = False
    env.norm_reward = False

    decision_rows: List[Dict[str, Any]] = []
    aircraft_rows: List[Dict[str, Any]] = []
    action_counts: Counter = Counter()
    prob_rows: List[np.ndarray] = []
    reward_sum = 0.0
    system_person_minutes = 0.0
    terminal_snapshot = None

    try:
        model = PPO.load(str(model_path), env=env, device=args.device)
        model.policy.set_training_mode(False)

        try:
            env.seed(int(args.eval_seed))
        except Exception:
            pass

        obs = env.reset()
        done = np.asarray([False])
        episode_steps = 0

        while not bool(done[0]):
            scenario = v3.mx.find_scenario(env)
            time_before = int(getattr(scenario, "time", episode_steps))
            system_person_minutes += v3.legacy.active_person_count(scenario)

            people_before = get_person_dict(scenario)
            finished_before = finished_set(scenario)
            q_before = queue_lengths(scenario)
            ac_before = aircraft_aggregate(scenario)

            pid = focal_pid(env)
            try:
                person = v3._focal_person(env, scenario)
            except Exception:
                person = None

            probs = np.asarray(v3.policy_probs(model, obs), dtype=float).reshape(-1)
            if probs.size != 4:
                raise RuntimeError(f"Expected 4 Joint probabilities, got shape {probs.shape}")

            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            passenger_action = ai // 2
            aircraft_action = ai % 2
            passenger_target = int(v3.CANDIDATES[passenger_action])
            aircraft_target = int(v3.CANDIDATES[aircraft_action])

            action_counts[ai] += 1
            prob_rows.append(probs.copy())

            p0 = project_candidate(scenario, spec, person, 0)
            p1 = project_candidate(scenario, spec, person, 1)

            obs, reward, done, infos = env.step(action)
            episode_steps += 1
            reward_scalar = fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            reward_sum += reward_scalar

            info = infos[0] if infos else {}
            if not isinstance(info, dict):
                info = {}
            if "terminal_snapshot" in info:
                terminal_snapshot = info["terminal_snapshot"]

            dispatched = bool(info.get("aircraft_dispatched", False))
            eligible_reported = int(info.get("aircraft_eligible_hub_count", ac_before["hub_eligible"]))
            if dispatched:
                dispatch_status = "SUCCESS"
            elif eligible_reported <= 0:
                dispatch_status = "NO_ELIGIBLE_HUB"
            else:
                dispatch_status = "REJECTED_WITH_ELIGIBLE"

            scenario_after = v3.mx.find_scenario(env)
            time_after = int(getattr(scenario_after, "time", time_before + 1))
            people_after = get_person_dict(scenario_after)
            finished_after = finished_set(scenario_after)
            q_after = queue_lengths(scenario_after)
            ac_after = aircraft_aggregate(scenario_after)

            generated_before = len(people_before)
            generated_after = len(people_after)
            nfinished_before = len(finished_before)
            nfinished_after = len(finished_after)

            row: Dict[str, Any] = {
                "step_index": episode_steps,
                "time_before": time_before,
                "time_after": time_after,
                "focal_pid": pid,
                "has_focal_passenger": bool(pid),
                "waiting_decisions_before": len(getattr(scenario, "waiting_decisions", []) or []),
                "joint_action": ai,
                "passenger_action": passenger_action,
                "passenger_target": passenger_target,
                "aircraft_action": aircraft_action,
                "aircraft_target": aircraft_target,
                "pair_consistent": int(passenger_action == aircraft_action),
                "prob_pair_00": probs[0],
                "prob_pair_01": probs[1],
                "prob_pair_10": probs[2],
                "prob_pair_11": probs[3],
                "prob_passenger_V0": probs[0] + probs[1],
                "prob_passenger_V1": probs[2] + probs[3],
                "prob_aircraft_V0": probs[0] + probs[2],
                "prob_aircraft_V1": probs[1] + probs[3],
                "chosen_prob": probs[ai],
                "policy_entropy": float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()),
                "reward": reward_scalar,
                "aircraft_dispatched": dispatched,
                "aircraft_eligible_reported": eligible_reported,
                "dispatch_status": dispatch_status,
                "generated_before": generated_before,
                "finished_before": nfinished_before,
                "backlog_before": generated_before - nfinished_before,
                "generated_after": generated_after,
                "finished_after": nfinished_after,
                "backlog_after": generated_after - nfinished_after,
                "queue_V0_before": int(q_before.get("0", 0)),
                "queue_V1_before": int(q_before.get("1", 0)),
                "queue_V2_before": int(q_before.get(str(v3.DESTINATION), 0)),
                "queue_V0_after": int(q_after.get("0", 0)),
                "queue_V1_after": int(q_after.get("1", 0)),
                "queue_V2_after": int(q_after.get(str(v3.DESTINATION), 0)),
                "queue_total_before": int(sum(q_before.values())),
                "queue_total_after": int(sum(q_after.values())),
                **p0,
                **p1,
            }

            for k, v in ac_before.items():
                row[f"{k}_before"] = v
            for k, v in ac_after.items():
                row[f"{k}_after"] = v

            decision_rows.append(row)

            if (
                episode_steps == 1
                or episode_steps % max(1, int(args.snapshot_every)) == 0
                or bool(done[0])
            ):
                aircraft_rows.extend(aircraft_snapshot_rows(scenario_after, time_after))

            if (
                episode_steps == 1
                or episode_steps % max(1, int(args.print_every)) == 0
                or bool(done[0])
            ):
                print(
                    f"[{args.cell} {int(args.step)//1000}k seed={args.eval_seed}] "
                    f"t={time_after:4d} action={ai} P->V{passenger_target} A->V{aircraft_target} "
                    f"dispatch={dispatch_status:<22} eligible={eligible_reported:2d} "
                    f"Q=({q_after.get('0',0):3d},{q_after.get('1',0):3d}) "
                    f"finished={nfinished_after:3d}/{generated_after:3d} "
                    f"backlog={generated_after-nfinished_after:3d}"
                )

            if episode_steps > int(args.max_time) + 5:
                raise RuntimeError("diagnostic exceeded hard guard")

        if terminal_snapshot is None:
            terminal_snapshot = v3.mx.snapshot_episode(v3.mx.find_scenario(env))

        metrics = v3.mx.metrics_from_terminal_snapshot(
            snapshot=terminal_snapshot,
            action_counts=action_counts,
            prob_rows=prob_rows,
            system_person_minutes=system_person_minutes,
            reward_sum=reward_sum,
            episode_steps=episode_steps,
        )

        windows, first_collapse = summarize_windows(
            decision_rows,
            window=int(args.window),
            collapse_no_eligible_threshold=float(args.collapse_no_eligible_threshold),
            collapse_backlog_threshold=int(args.collapse_backlog_threshold),
        )

        statuses = Counter(r["dispatch_status"] for r in decision_rows)
        pairs = Counter(int(r["joint_action"]) for r in decision_rows)
        ps = Counter(int(r["passenger_action"]) for r in decision_rows)
        aa = Counter(int(r["aircraft_action"]) for r in decision_rows)

        final_completion = fnum(metrics.get("completion_rate"), 0.0)
        formal_att = fnum(metrics.get("ATT")) if final_completion >= 0.999999 else float("nan")
        formal_awt = fnum(metrics.get("AWT")) if final_completion >= 0.999999 else float("nan")

        summary = {
            "cell": args.cell,
            "env_key": env_key,
            "joint_arch_name": v4.JOINT_ARCH_NAMES_V4.get(env_key, env_key),
            "method_id": method_id,
            "train_step": int(args.step),
            "eval_seed": int(args.eval_seed),
            "model_source": source_kind,
            "model_path": str(model_path),
            "vecnormalize_path": str(vec_path),
            "max_time": int(args.max_time),
            "episode_steps": episode_steps,
            "completion_rate": final_completion,
            "valid_full_completion": final_completion >= 0.999999,
            "formal_ATT": formal_att,
            "formal_AWT": formal_awt,
            "system_person_minutes_per_passenger": fnum(
                metrics.get("system_person_minutes_per_passenger")
            ),
            "reward_sum": reward_sum,
            "pair_action_counts": dict(pairs),
            "pair_action_shares": {str(k): v / max(1, episode_steps) for k, v in pairs.items()},
            "passenger_action_shares": {
                "V0": ps[0] / max(1, episode_steps),
                "V1": ps[1] / max(1, episode_steps),
            },
            "aircraft_action_shares": {
                "V0": aa[0] / max(1, episode_steps),
                "V1": aa[1] / max(1, episode_steps),
            },
            "dispatch_counts": dict(statuses),
            "dispatch_success_rate": statuses["SUCCESS"] / max(1, episode_steps),
            "dispatch_no_eligible_rate": statuses["NO_ELIGIBLE_HUB"] / max(1, episode_steps),
            "dispatch_rejected_with_eligible_rate": (
                statuses["REJECTED_WITH_ELIGIBLE"] / max(1, episode_steps)
            ),
            "max_queue_V0": max([int(r["queue_V0_after"]) for r in decision_rows] or [0]),
            "max_queue_V1": max([int(r["queue_V1_after"]) for r in decision_rows] or [0]),
            "max_backlog": max([int(r["backlog_after"]) for r in decision_rows] or [0]),
            "candidate_collapse_onset": first_collapse,
            "collapse_rule": {
                "window_steps": int(args.window),
                "dispatch_no_eligible_rate_ge": float(args.collapse_no_eligible_threshold),
                "backlog_end_ge": int(args.collapse_backlog_threshold),
                "note": "Diagnostic marker only; not a scientific definition of collapse.",
            },
            "formal_metrics_raw": metrics,
            "important_semantics": (
                "NO_ELIGIBLE_HUB means the Joint wrapper had zero IDLE, passenger-free, "
                "non-turnaround aircraft at hub V2 before attempting reposition. "
                "It does NOT by itself prove that the selected target was physically invalid."
            ),
        }

        write_csv(out_dir / "decision_trace.csv", decision_rows)
        write_csv(out_dir / "window_summary.csv", windows)
        write_csv(out_dir / "aircraft_snapshots.csv", aircraft_rows)
        write_json(out_dir / "summary.json", summary)

        # A concise human-readable text summary.
        lines = [
            f"cell={args.cell}",
            f"step={args.step}",
            f"seed={args.eval_seed}",
            f"completion={final_completion:.6f}",
            f"formal_ATT={formal_att}",
            f"formal_AWT={formal_awt}",
            f"dispatch_success_rate={summary['dispatch_success_rate']:.6f}",
            f"dispatch_no_eligible_rate={summary['dispatch_no_eligible_rate']:.6f}",
            f"dispatch_rejected_with_eligible_rate={summary['dispatch_rejected_with_eligible_rate']:.6f}",
            f"passenger_action_shares={summary['passenger_action_shares']}",
            f"aircraft_action_shares={summary['aircraft_action_shares']}",
            f"pair_action_shares={summary['pair_action_shares']}",
            f"max_queue_V0={summary['max_queue_V0']}",
            f"max_queue_V1={summary['max_queue_V1']}",
            f"max_backlog={summary['max_backlog']}",
            f"candidate_collapse_onset={first_collapse}",
        ]
        (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        zpath = zip_outputs(out_dir)

        print("\n" + "=" * 110)
        print("JOINT COLLAPSE TRACE COMPLETE")
        print("=" * 110)
        print(f"cell                     : {args.cell}")
        print(f"checkpoint               : {args.step}")
        print(f"seed                     : {args.eval_seed}")
        print(f"completion               : {final_completion:.3%}")
        print(f"formal ATT               : {formal_att}")
        print(f"dispatch success         : {summary['dispatch_success_rate']:.3%}")
        print(f"dispatch NO eligible hub : {summary['dispatch_no_eligible_rate']:.3%}")
        print(f"dispatch rejected w/elig : {summary['dispatch_rejected_with_eligible_rate']:.3%}")
        print(f"passenger shares         : {summary['passenger_action_shares']}")
        print(f"aircraft shares          : {summary['aircraft_action_shares']}")
        print(f"pair shares              : {summary['pair_action_shares']}")
        print(f"max queues V0/V1         : {summary['max_queue_V0']} / {summary['max_queue_V1']}")
        print(f"candidate collapse onset : {first_collapse}")
        print(f"output                   : {out_dir}")
        print(f"upload zip               : {zpath}")
        return 0

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            v3.core.restore_process_patches()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detailed deterministic Joint-policy collapse trace for the V4 60M UAM matrix."
    )
    p.add_argument("--run-root", required=True)
    p.add_argument("--cell", required=True, help="e.g. J0__CURRENT")
    p.add_argument("--step", type=int, default=600000)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--max-time", type=int, default=2500)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--collapse-no-eligible-threshold", type=float, default=0.80)
    p.add_argument("--collapse-backlog-threshold", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
