# -*- coding: utf-8 -*-
"""
Auto post-training test for UAGMC E3/E4/E5
==========================================

Scans the newest run from train_uagmc_E3_E4_E5_serial_1m.py.

A stage is evaluated ONLY if BOTH formal 1M files exist:
    checkpoints/uam_ppo_1000000_steps.zip
    checkpoints/uam_ppo_vecnormalize_1000000_steps.pkl

So if E3/E4 are finished and E5 is still training, E5 is automatically
SKIPPED instead of treated as an error.

For every completed stage, default evaluation checks all available:
    50k, 100k, ..., 1M
with eval seeds 123,124,125 on the same fixed passenger trace.

Metrics:
    ATT, AWT, access, flight, completion, backlog
    P50/P90/P95/max travel time
    V0/V1 deterministic action share
    policy probability / normalized entropy / margin
    final queues
    fixed-fleet counts
    E3/E4/E5 physical counters if exposed by the training environment

NO TRAINING happens here.
The exact E3/E4/E5 environment patches are imported from:
    train_uagmc_E3_E4_E5_serial_1m.py

Run:
    python evaluate_uagmc_E3_E4_E5_auto.py

Fast final-only:
    python evaluate_uagmc_E3_E4_E5_auto.py --final-only

Output:
    <run-root>/posttrain_E3_E4_E5/
        stage_scan.csv
        checkpoint_inventory.csv
        episode_metrics.csv
        curve_by_stage.csv
        final_1m_comparison.csv
        skipped_stages.csv
        errors.csv
        summary.txt
        UPLOAD_THIS_E3_E4_E5_analysis.zip
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import random
import re
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    trainmod = importlib.import_module("train_uagmc_E3_E4_E5_serial_1m")
except Exception as exc:
    raise RuntimeError(
        "Cannot import train_uagmc_E3_E4_E5_serial_1m.py. "
        "Put this evaluator beside the training script in UAGMC-main."
    ) from exc


STAGES = ("E3_SINGLE_PAX", "E4_TURNAROUND", "E5_PAD")
FINAL_STEP = 1_000_000
DEFAULT_EVAL_EVERY = 50_000


def fnum(x: Any, default=float("nan")) -> float:
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def fmean(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def fstd(xs: Iterable[Any]) -> float:
    a = np.asarray([fnum(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return 0.0 if len(a) == 1 else float(a.std(ddof=1))


def parse_ints(s: str) -> List[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    return x


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {}
            for k, v in r.items():
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    out[k] = json.dumps(jsonable(v), ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def auto_run_root() -> Path:
    serial = ROOT / "serial_runs"
    found = [
        p for p in serial.glob("uagmc_E3_E4_E5_LQ_1m_seed*_*")
        if p.is_dir()
    ]
    if not found:
        found = [
            p for p in serial.glob("*E3_E4_E5*LQ*")
            if p.is_dir()
        ]
    if not found:
        raise FileNotFoundError(
            "Cannot auto-detect E3/E4/E5 run. Use --run-root."
        )
    return max(found, key=lambda p: p.stat().st_mtime).resolve()


@dataclass(frozen=True)
class StageStatus:
    stage: str
    completed: bool
    reason: str
    stage_dir: Path


@dataclass(frozen=True)
class Spec:
    stage: str
    step: int
    model: Path
    vec: Path


def scan_stages(run_root: Path) -> List[StageStatus]:
    out = []
    for stage in STAGES:
        d = run_root / stage
        ck = d / "checkpoints"
        m = ck / f"uam_ppo_{FINAL_STEP}_steps.zip"
        v = ck / f"uam_ppo_vecnormalize_{FINAL_STEP}_steps.pkl"
        if not d.exists():
            ok, reason = False, "directory missing"
        elif not m.exists():
            ok, reason = False, "1M model missing"
        elif not v.exists():
            ok, reason = False, "1M VecNormalize missing"
        else:
            ok, reason = True, "1M model + VecNormalize found"
        out.append(StageStatus(stage, ok, reason, d))
    return out


def discover(status: StageStatus, every: int, final_only: bool) -> List[Spec]:
    if not status.completed:
        return []
    ck = status.stage_dir / "checkpoints"
    out = []
    for model in ck.glob("uam_ppo_*_steps.zip"):
        m = re.search(r"uam_ppo_(\d+)_steps", model.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > FINAL_STEP or step % every:
            continue
        if final_only and step != FINAL_STEP:
            continue
        vec = ck / f"uam_ppo_vecnormalize_{step}_steps.pkl"
        if vec.exists():
            out.append(Spec(status.stage, step, model.resolve(), vec.resolve()))
    return sorted(out, key=lambda x: (STAGES.index(x.stage), x.step))


def find_scenario(obj: Any):
    if hasattr(obj, "venv"):
        obj = obj.venv
    if hasattr(obj, "envs") and obj.envs:
        obj = obj.envs[0]
    seen = set()
    for _ in range(60):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        if all(hasattr(obj, k) for k in ("persons", "person_travel_records", "finished_ids")):
            return obj
        if hasattr(obj, "scenario"):
            sc = getattr(obj, "scenario")
            if sc is not None and all(hasattr(sc, k) for k in ("persons", "person_travel_records", "finished_ids")):
                return sc
        if hasattr(obj, "env") and getattr(obj, "env") is not obj:
            obj = obj.env
            continue
        if hasattr(obj, "unwrapped") and obj.unwrapped is not obj:
            obj = obj.unwrapped
            continue
        break
    raise RuntimeError(f"Cannot locate Scenario; stopped at {type(obj)}")


def waiting_count(obj: Any):
    seen = set()
    for _ in range(60):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        state = getattr(obj, "state", None)
        if isinstance(state, dict) and "waiting_decisions" in state:
            return len(state.get("waiting_decisions") or [])
        if hasattr(obj, "env") and getattr(obj, "env") is not obj:
            obj = obj.env
            continue
        if hasattr(obj, "unwrapped") and obj.unwrapped is not obj:
            obj = obj.unwrapped
            continue
        break
    return None


class TerminalCapture(gym.Wrapper):
    def snapshot(self):
        sc = find_scenario(self.env)
        pobj = getattr(sc, "persons", None)
        persons = getattr(pobj, "persons", {}) if pobj is not None else {}
        persons = persons or {}
        records = getattr(sc, "person_travel_records", {}) or {}
        finished = set(getattr(sc, "finished_ids", []) or [])
        finished_s = {str(x) for x in finished}
        rows = []
        for pid_raw, p in persons.items():
            pid = str(pid_raw)
            recs = records.get(pid_raw) or records.get(pid) or []
            rec = recs[-1] if recs else {}
            st, et = rec.get("start_time"), rec.get("end_time")
            travel = float("nan")
            if st is not None and et is not None:
                try:
                    travel = float(et) - float(st)
                except Exception:
                    pass
            stats = getattr(p, "time_stats", {}) or {}
            rows.append({
                "pid": pid,
                "finished": bool(
                    pid_raw in finished or pid in finished_s or et is not None
                    or str(getattr(p, "state", "")).lower() == "finished"
                ),
                "travel": travel,
                "access": fnum(stats.get("to_vertiport", np.nan)),
                "wait": fnum(stats.get("wait_uam", np.nan)),
                "fly": fnum(stats.get("fly", np.nan)),
            })
        queues = {}
        try:
            for vid in (0, 1):
                vp = sc.vertiports.vertiport_list[str(vid)]
                queues[str(vid)] = len(list(getattr(vp, "person_list", []) or []))
        except Exception:
            pass
        diag = {}
        if hasattr(sc, "get_fixed_fleet_diagnostics"):
            try:
                diag = sc.get_fixed_fleet_diagnostics() or {}
            except Exception:
                pass
        return {
            "rows": rows,
            "n": len(persons),
            "queues": queues,
            "diag": diag,
            "stage_stats": dict(getattr(sc, "_e345_stats", {}) or {}),
        }

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            if bool(term) or bool(trunc):
                info = dict(info)
                info["terminal_snapshot"] = self.snapshot()
            return obs, reward, term, trunc, info
        obs, reward, done, info = out
        if bool(done):
            info = dict(info)
            info["terminal_snapshot"] = self.snapshot()
        return obs, reward, done, info


def build_eval_env(stage: str, fleet_size: int, vec: Path, max_time: int, monitor: Path):
    trainmod.restore_source_methods()
    trainmod.install_stage_patch(stage)

    factory = trainmod.make_stage_env_factory(
        stage=stage,
        fleet_size=int(fleet_size),
        env_index=9999,
        run_dir=monitor,
        max_time=int(max_time),
    )

    def _make():
        trainmod.restore_source_methods()
        trainmod.install_stage_patch(stage)
        return TerminalCapture(factory())

    raw = DummyVecEnv([_make])
    env = VecNormalize.load(str(vec), raw)
    env.training = False
    env.norm_reward = False
    return env


def probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        ot, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(ot).distribution
        p = dist.probs if getattr(dist, "probs", None) is not None else torch.softmax(dist.logits, dim=-1)
        return p.detach().cpu().numpy().reshape(-1, 2)[0]


def summarize_snapshot(snap, decisions, all_actions, pvals, reward_sum, ep_steps, gate_seen):
    completed = [
        r for r in snap["rows"]
        if r["finished"] and np.isfinite(fnum(r["travel"]))
    ]
    travel = np.asarray([fnum(r["travel"]) for r in completed], dtype=float)
    travel = travel[np.isfinite(travel)]

    def comp(key):
        a = np.asarray([fnum(r[key]) for r in completed], dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else float("nan")

    att = float(travel.mean()) if len(travel) else float("nan")
    access, awt, aft = comp("access"), comp("wait"), comp("fly")
    n, nf = int(snap["n"]), len(completed)
    dden, aden = max(1, sum(decisions.values())), max(1, sum(all_actions.values()))

    out = {
        "ATT": att,
        "AGT_access": access,
        "AWT": awt,
        "AFT": aft,
        "travel_median": float(np.median(travel)) if len(travel) else float("nan"),
        "travel_p90": float(np.percentile(travel, 90)) if len(travel) else float("nan"),
        "travel_p95": float(np.percentile(travel, 95)) if len(travel) else float("nan"),
        "travel_max": float(np.max(travel)) if len(travel) else float("nan"),
        "N": n,
        "N_finished": nf,
        "completion_rate": nf / n if n else float("nan"),
        "final_backlog": n - nf,
        "episode_reward": reward_sum,
        "episode_steps": ep_steps,
        "decision_gate_found": gate_seen,
        "decision_action_v0_share": decisions.get(0, 0) / dden,
        "decision_action_v1_share": decisions.get(1, 0) / dden,
        "allstep_action_v0_share": all_actions.get(0, 0) / aden,
        "allstep_action_v1_share": all_actions.get(1, 0) / aden,
        "final_queue_v0": int(snap.get("queues", {}).get("0", 0)),
        "final_queue_v1": int(snap.get("queues", {}).get("1", 0)),
    }

    if pvals:
        pa = np.vstack(pvals)
        ent = -np.sum(pa * np.log(np.clip(pa, 1e-12, 1.0)), axis=1)
        out.update({
            "mean_policy_prob_v0": float(pa[:, 0].mean()),
            "mean_policy_prob_v1": float(pa[:, 1].mean()),
            "policy_entropy_normalized": float(ent.mean() / math.log(2.0)),
            "mean_policy_margin": float(np.abs(pa[:, 0] - pa[:, 1]).mean()),
            "mean_policy_max_prob": float(np.max(pa, axis=1).mean()),
        })
    else:
        for k in ("mean_policy_prob_v0", "mean_policy_prob_v1",
                  "policy_entropy_normalized", "mean_policy_margin",
                  "mean_policy_max_prob"):
            out[k] = float("nan")

    diag = snap.get("diag", {}) or {}
    for k in ("fleet_size", "service_arrivals", "reposition_departures", "reposition_arrivals"):
        out[k] = diag.get(k, np.nan)

    stats = snap.get("stage_stats", {}) or {}
    for k in (
        "single_pax_service_departures",
        "turnaround_starts",
        "turnaround_releases",
        "pad_departure_reservations",
        "pad_landing_reservations",
        "service_pad_blocks",
        "reposition_pad_blocks",
    ):
        out[k] = stats.get(k, 0)

    return out


def run_episode(spec: Spec, fleet_size: int, eval_seed: int, max_time: int, device: str, monitor: Path):
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    env = build_eval_env(spec.stage, fleet_size, spec.vec, max_time, monitor)
    try:
        model = PPO.load(str(spec.model), env=env, device=device)
        model.policy.set_training_mode(False)
        try:
            env.seed(eval_seed)
        except Exception:
            pass

        obs = env.reset()
        raw = env.venv.envs[0]
        done = np.array([False])
        decisions, all_actions = Counter(), Counter()
        pvals = []
        reward_sum = 0.0
        ep_steps = 0
        gate_seen = False
        snap = None

        while not bool(done[0]):
            wc = waiting_count(raw)
            if wc is not None:
                gate_seen = True

            pp = probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            ai = int(np.asarray(action).reshape(-1)[0])
            all_actions[ai] += 1
            if wc is not None and wc > 0:
                decisions[ai] += 1
                pvals.append(pp.copy())

            obs, reward, done, infos = env.step(action)
            reward_sum += fnum(np.asarray(reward).reshape(-1)[0], 0.0)
            ep_steps += 1

            if infos and isinstance(infos[0], dict) and "terminal_snapshot" in infos[0]:
                snap = infos[0]["terminal_snapshot"]

            if ep_steps > max_time + 100:
                raise RuntimeError("episode exceeded max-time guard")

        if snap is None:
            raise RuntimeError("terminal snapshot not captured")

        if not gate_seen:
            decisions = Counter(all_actions)
            pvals = []

        row = summarize_snapshot(
            snap, decisions, all_actions, pvals, reward_sum, ep_steps, gate_seen
        )
        row.update({
            "stage": spec.stage,
            "train_step": spec.step,
            "eval_seed": eval_seed,
            "fleet_size_selected": fleet_size,
            "model_path": str(spec.model),
            "vecnormalize_path": str(spec.vec),
        })
        return row

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            trainmod.restore_source_methods()
        except Exception:
            pass
        gc.collect()


METRICS = (
    "ATT", "AGT_access", "AWT", "AFT", "travel_median", "travel_p90",
    "travel_p95", "travel_max", "completion_rate", "final_backlog",
    "episode_reward", "decision_action_v0_share", "decision_action_v1_share",
    "mean_policy_prob_v0", "mean_policy_prob_v1",
    "policy_entropy_normalized", "mean_policy_margin", "mean_policy_max_prob",
    "final_queue_v0", "final_queue_v1", "service_arrivals",
    "reposition_departures", "reposition_arrivals",
    "single_pax_service_departures", "turnaround_starts",
    "turnaround_releases", "pad_departure_reservations",
    "pad_landing_reservations", "service_pad_blocks", "reposition_pad_blocks",
)


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["stage"], int(r["train_step"]))].append(r)
    out = []
    for (stage, step), group in sorted(
        groups.items(), key=lambda x: (STAGES.index(x[0][0]), x[0][1])
    ):
        row = {
            "stage": stage,
            "train_step": step,
            "n_eval_seeds": len(group),
            "fleet_size": int(group[0]["fleet_size_selected"]),
        }
        for m in METRICS:
            vals = [r.get(m, np.nan) for r in group]
            row[m + "_mean"] = fmean(vals)
            row[m + "_std"] = fstd(vals)
        out.append(row)
    return out


def make_summary(path: Path, run_root: Path, statuses, curve, final_rows, errors):
    lines = [
        "=" * 118,
        "UAGMC E3/E4/E5 AUTO POST-TRAINING TEST",
        "=" * 118,
        f"Run root: {run_root}",
        "",
        "STAGE SCAN",
        "-" * 118,
    ]
    for s in statuses:
        lines.append(
            f"{s.stage:<20} | {'EVALUATE' if s.completed else 'SKIP':<8} | {s.reason}"
        )

    lines += ["", "FINAL 1M COMPARISON", "-" * 118]
    for r in final_rows:
        lines.append(
            f"{r['stage']:<20} | ATT={fnum(r.get('ATT_mean')):.3f} | "
            f"AWT={fnum(r.get('AWT_mean')):.3f} | "
            f"finish={100*fnum(r.get('completion_rate_mean')):.2f}% | "
            f"V0={100*fnum(r.get('decision_action_v0_share_mean')):.1f}% | "
            f"H={fnum(r.get('policy_entropy_normalized_mean')):.3f}"
        )

    lines += [
        "",
        "GUARDRAILS",
        "-" * 118,
        "Unfinished stages are skipped completely.",
        "ATT must be read together with completion/backlog.",
        "123/124/125 can be identical under deterministic trace/policy.",
        "This is one training seed; it is a development comparison, not final statistics.",
        f"errors={len(errors)}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_zip(outdir: Path) -> Path:
    zpath = outdir / "UPLOAD_THIS_E3_E4_E5_analysis.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob("*")):
            if not p.is_file() or p == zpath:
                continue
            if p.suffix.lower() in {".csv", ".json", ".txt"}:
                z.write(p, p.relative_to(outdir).as_posix())
    return zpath


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default=None)
    p.add_argument("--eval-seeds", default="123,124,125")
    p.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    p.add_argument("--max-time", type=int, default=600)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--final-only", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    run_root = Path(args.run_root).expanduser() if args.run_root else auto_run_root()
    if not run_root.is_absolute():
        run_root = (ROOT / run_root).resolve()
    else:
        run_root = run_root.resolve()

    manifest_path = run_root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fleet_size = int(
        manifest.get(
            "fleet_size_frozen_for_all_stages",
            manifest.get("fleet_size", -1),
        )
    )
    if fleet_size <= 0:
        raise RuntimeError("Cannot resolve frozen fleet size from experiment manifest")

    device = "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    eval_seeds = parse_ints(args.eval_seeds)
    statuses = scan_stages(run_root)

    outdir = run_root / "posttrain_E3_E4_E5"
    outdir.mkdir(parents=True, exist_ok=True)
    monitor = outdir / "_monitor"
    monitor.mkdir(parents=True, exist_ok=True)

    scan_rows = [{
        "stage": s.stage,
        "completed": s.completed,
        "action": "EVALUATE" if s.completed else "SKIP",
        "reason": s.reason,
    } for s in statuses]
    write_csv(outdir / "stage_scan.csv", scan_rows)
    write_csv(
        outdir / "skipped_stages.csv",
        [r for r in scan_rows if not r["completed"]],
    )

    completed = [s for s in statuses if s.completed]
    specs = []
    for s in completed:
        specs.extend(discover(s, int(args.eval_every), bool(args.final_only)))

    write_csv(
        outdir / "checkpoint_inventory.csv",
        [{
            "stage": x.stage,
            "train_step": x.step,
            "model_path": str(x.model),
            "vecnormalize_path": str(x.vec),
        } for x in specs],
    )

    print("=" * 118)
    print("UAGMC E3/E4/E5 AUTO TEST")
    print("=" * 118)
    print(f"Run root : {run_root}")
    print(f"Fleet    : {fleet_size}")
    for s in statuses:
        print(
            f"{s.stage:<20} | {'EVALUATE' if s.completed else 'SKIP':<8} | {s.reason}"
        )
    print("=" * 118)

    if not specs:
        print("No completed stage to evaluate yet.")
        return 0

    rows, errors = [], []
    total = len(specs) * len(eval_seeds)
    job = 0

    for spec in specs:
        for es in eval_seeds:
            job += 1
            print(
                f"[{job:>3}/{total}] {spec.stage:<20} "
                f"step={spec.step:>9,d} eval_seed={es}",
                flush=True,
            )
            try:
                row = run_episode(
                    spec, fleet_size, es, int(args.max_time), device, monitor
                )
                rows.append(row)
                print(
                    f"    ATT={row['ATT']:.3f} | AWT={row['AWT']:.3f} | "
                    f"finish={row['N_finished']}/{row['N']} | "
                    f"V0={100*row['decision_action_v0_share']:.1f}% | "
                    f"H={row['policy_entropy_normalized']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                err = {
                    "stage": spec.stage,
                    "train_step": spec.step,
                    "eval_seed": es,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                print(f"    ERROR: {repr(exc)}", flush=True)
                if args.fail_fast:
                    write_csv(outdir / "errors.csv", errors)
                    raise

    write_csv(outdir / "episode_metrics.csv", rows)
    write_csv(outdir / "errors.csv", errors)

    curve = aggregate(rows)
    write_csv(outdir / "curve_by_stage.csv", curve)

    final_rows = [r for r in curve if int(r["train_step"]) == FINAL_STEP]
    write_csv(outdir / "final_1m_comparison.csv", final_rows)

    for name in (
        "experiment_manifest.json",
        "fleet_calibration.csv",
        "fleet_calibration_summary.json",
        "serial_status.csv",
        "experiment_end.json",
    ):
        src = run_root / name
        if src.exists():
            shutil.copy2(src, outdir / ("training_" + name))

    for s in completed:
        for name in ("preflight.json", "run_manifest.json", "run_end.json", "training_milestones.csv"):
            src = s.stage_dir / name
            if src.exists():
                shutil.copy2(src, outdir / f"{s.stage}_{name}")

    make_summary(
        outdir / "summary.txt",
        run_root,
        statuses,
        curve,
        final_rows,
        errors,
    )

    write_json(
        outdir / "analysis_manifest.json",
        {
            "run_root": str(run_root),
            "fleet_size": fleet_size,
            "completed_stages": [s.stage for s in completed],
            "skipped_stages": [s.stage for s in statuses if not s.completed],
            "eval_seeds": eval_seeds,
            "max_time": int(args.max_time),
            "final_only": bool(args.final_only),
            "reuses_exact_training_stage_patch": True,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    bundle = build_zip(outdir)

    print("\n" + "=" * 118)
    print("AUTO TEST COMPLETE")
    print("=" * 118)
    print(f"Stage scan : {outdir / 'stage_scan.csv'}")
    print(f"Curve      : {outdir / 'curve_by_stage.csv'}")
    print(f"Final 1M   : {outdir / 'final_1m_comparison.csv'}")
    print(f"Summary    : {outdir / 'summary.txt'}")
    print("-" * 118)
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(bundle)
    print("=" * 118)

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
