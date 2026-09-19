#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-training evaluator for the 1.6M reposition experiment:
  - longest_queue:    800k
  - vertisync_simple: 800k

This file does NOT train. It evaluates saved checkpoints only.

It intentionally imports:
    train_uagmc_reposition_lq_vs_vertisync_800k.py
and reuses its EXACT reposition monkey-patch, so evaluation cannot silently
implement a different LQ / VertiSync-simple rule.

Default:
  checkpoints : 50k,100k,...,800k
  eval seeds  : 123,124,125
  trace       : train_data/passengers_300.csv
  max_time    : 600
  deterministic policy
  inference   : CPU

Outputs under <run-root>/posttrain_analysis/:
  episode_metrics.csv
  curve_by_method.csv
  final_800k_comparison.csv
  return_home_reference_800k.csv   (if previous run is found)
  threeway_final_comparison.csv
  errors.csv
  summary.txt
  att_curve.png / awt_curve.png / completion_curve.png / action_share_curve.png
  UPLOAD_THIS_reposition_1600k_analysis.zip

Usage:
  python evaluate_uagmc_reposition_1600k.py

Quick final-only check:
  python evaluate_uagmc_reposition_1600k.py --final-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    import train_uagmc_reposition_lq_vs_vertisync_800k as trainmod
except Exception as exc:
    raise RuntimeError(
        "Put train_uagmc_reposition_lq_vs_vertisync_800k.py beside this evaluator."
    ) from exc

from utilss.make_env_fleet import make_env
from at_obj.scenario_fixed_fleet import ConservedFleetScenario

ORIGINAL_RETURN_HOME = ConservedFleetScenario._dispatch_fixed_returns
METHODS = ("longest_queue", "vertisync_simple")
FLEET_SIZE = 16
CANDIDATES = [0, 1]
TO_VERTIPORT = 2
PASSENGER_FILE = ROOT / "train_data" / "passengers_300.csv"


def ffloat(x, default=float("nan")):
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return default


def mean(xs: Iterable[Any]) -> float:
    a = np.asarray([ffloat(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def std(xs: Iterable[Any]) -> float:
    a = np.asarray([ffloat(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    if len(a) == 1:
        return 0.0
    return float(a.std(ddof=1))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]):
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
                    out[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    out[k] = v
            w.writerow(out)


def auto_run_root() -> Path:
    serial = ROOT / "serial_runs"
    cand = list(serial.glob("uagmc_reposition_LQ_vs_VertiSync_N16_seed*_800k_*"))
    cand = [p for p in cand if p.is_dir()]
    if not cand:
        raise FileNotFoundError(
            "Cannot auto-find reposition run. Pass --run-root explicitly."
        )
    return max(cand, key=lambda p: p.stat().st_mtime).resolve()


def find_scenario(env):
    obj, seen = env, set()
    for _ in range(60):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        if all(hasattr(obj, x) for x in ("person_travel_records", "persons", "finished_ids")):
            return obj
        if hasattr(obj, "scenario"):
            sc = getattr(obj, "scenario")
            if sc is not None and all(hasattr(sc, x) for x in ("person_travel_records", "persons", "finished_ids")):
                return sc
        if hasattr(obj, "env"):
            nxt = getattr(obj, "env")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue
        if hasattr(obj, "unwrapped"):
            nxt = getattr(obj, "unwrapped")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue
        break
    raise RuntimeError("Cannot locate Scenario")


def waiting_decisions(env) -> Optional[int]:
    obj, seen = env, set()
    for _ in range(50):
        if id(obj) in seen:
            break
        seen.add(id(obj))
        state = getattr(obj, "state", None)
        if isinstance(state, dict) and "waiting_decisions" in state:
            try:
                return len(state.get("waiting_decisions") or [])
            except Exception:
                pass
        if hasattr(obj, "waiting_decisions"):
            try:
                return len(getattr(obj, "waiting_decisions") or [])
            except Exception:
                pass
        if hasattr(obj, "env"):
            nxt = getattr(obj, "env")
            if nxt is not None and nxt is not obj:
                obj = nxt
                continue
        break
    return None


class TerminalCapture(gym.Wrapper):
    def _snapshot(self):
        sc = find_scenario(self.env)
        persons_obj = getattr(sc, "persons", None)
        persons = getattr(persons_obj, "persons", {}) if persons_obj else {}
        persons = persons or {}
        records = getattr(sc, "person_travel_records", {}) or {}
        finished = set(getattr(sc, "finished_ids", []) or [])
        finished_s = {str(x) for x in finished}
        rows = []
        for pid_raw, p in persons.items():
            pid = str(pid_raw)
            recs = records.get(pid_raw) or records.get(pid) or []
            rec = recs[-1] if recs else {}
            start, end = rec.get("start_time"), rec.get("end_time")
            travel = float("nan")
            if start is not None and end is not None:
                try:
                    travel = float(end) - float(start)
                except Exception:
                    pass
            stats = getattr(p, "time_stats", {}) or {}
            rows.append({
                "pid": pid,
                "finished": bool(pid_raw in finished or pid in finished_s or end is not None),
                "travel": travel,
                "access": ffloat(stats.get("to_vertiport", np.nan)),
                "wait": ffloat(stats.get("wait_uam", np.nan)),
                "fly": ffloat(stats.get("fly", np.nan)),
            })
        queues = {}
        try:
            for vid in CANDIDATES:
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
        return {"rows": rows, "n": len(persons), "queues": queues, "diag": diag}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, rew, term, trunc, info = out
            if bool(term) or bool(trunc):
                info = dict(info)
                info["terminal_snapshot"] = self._snapshot()
            return obs, rew, term, trunc, info
        obs, rew, done, info = out
        if bool(done):
            info = dict(info)
            info["terminal_snapshot"] = self._snapshot()
        return obs, rew, done, info


def install_rule(mode: str):
    if mode == "return_home":
        ConservedFleetScenario._dispatch_fixed_returns = ORIGINAL_RETURN_HOME
    else:
        trainmod.install_reposition_patch(mode)


def build_env(mode: str, vec_path: Path, max_time: int, outdir: Path):
    install_rule(mode)
    def _make():
        install_rule(mode)
        base = make_env(
            max_time=max_time,
            log_dir=outdir / "_monitor",
            env_index=0,
            person_spawn_file=str(PASSENGER_FILE),
            candidate_from_vertiports=CANDIDATES,
            to_vertiport=TO_VERTIPORT,
            enable_logger=False,
            fleet_mode="conserved_closed_loop",
            fleet_size=FLEET_SIZE,
            fleet_assertions=True,
        )()
        return TerminalCapture(base)
    raw = DummyVecEnv([_make])
    env = VecNormalize.load(str(vec_path), raw)
    env.training = False
    env.norm_reward = False
    return env


def raw_env(vec):
    obj = vec.venv if hasattr(vec, "venv") else vec
    return obj.envs[0] if hasattr(obj, "envs") and obj.envs else obj


def probs(model: PPO, obs) -> np.ndarray:
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs)
        d = model.policy.get_distribution(t).distribution
        p = d.probs if getattr(d, "probs", None) is not None else torch.softmax(d.logits, dim=-1)
        return np.asarray(p.detach().cpu().numpy(), dtype=float).reshape(-1, 2)[0]


def evaluate_one(method: str, step: int, model_path: Path, vec_path: Path,
                 eval_seed: int, max_time: int, device: str, outdir: Path) -> Dict[str, Any]:
    seed_all(eval_seed)
    env = build_env(method, vec_path, max_time, outdir)
    try:
        model = PPO.load(str(model_path), env=env, device=device)
        model.policy.set_training_mode(False)
        try:
            env.seed(eval_seed)
        except Exception:
            pass
        obs = env.reset()
        done = np.array([False], dtype=bool)
        base = raw_env(env)
        decision_actions, all_actions = Counter(), Counter()
        decision_probs: List[np.ndarray] = []
        gate_found = False
        snapshot = None
        total_reward = 0.0
        steps = 0
        while not bool(done[0]):
            waiting = waiting_decisions(base)
            if waiting is not None:
                gate_found = True
            pp = probs(model, obs)
            action, _ = model.predict(obs, deterministic=True)
            a = int(np.asarray(action).reshape(-1)[0])
            all_actions[a] += 1
            if waiting is not None and waiting > 0:
                decision_actions[a] += 1
                decision_probs.append(pp.copy())
            obs, rew, done, infos = env.step(action)
            total_reward += ffloat(np.asarray(rew).reshape(-1)[0], 0.0)
            steps += 1
            if infos and isinstance(infos[0], dict) and "terminal_snapshot" in infos[0]:
                snapshot = infos[0]["terminal_snapshot"]
            if steps > max_time + 100:
                raise RuntimeError("episode exceeded horizon guard")
        if snapshot is None:
            raise RuntimeError("terminal snapshot missing")
        if not gate_found:
            decision_actions = Counter(all_actions)
            decision_probs = []

        rows = snapshot["rows"]
        completed = [r for r in rows if r["finished"] and np.isfinite(ffloat(r["travel"]))]
        tr = np.asarray([ffloat(r["travel"]) for r in completed], dtype=float)
        def component(k):
            a = np.asarray([ffloat(r[k]) for r in completed], dtype=float)
            a = a[np.isfinite(a)]
            return float(a.mean()) if len(a) else float("nan")
        att = float(tr.mean()) if len(tr) else float("nan")
        access, awt, aft = component("access"), component("wait"), component("fly")
        n = int(snapshot["n"])
        nf = len(completed)
        dd = max(1, sum(decision_actions.values()))
        result = {
            "method": method,
            "train_step": step,
            "eval_seed": eval_seed,
            "ATT": att,
            "AGT_access": access,
            "AWT": awt,
            "AFT": aft,
            "completion_rate": nf / n if n else float("nan"),
            "N_finished": nf,
            "N": n,
            "final_backlog": n - nf,
            "travel_median": float(np.median(tr)) if len(tr) else float("nan"),
            "travel_p90": float(np.percentile(tr, 90)) if len(tr) else float("nan"),
            "travel_p95": float(np.percentile(tr, 95)) if len(tr) else float("nan"),
            "travel_max": float(np.max(tr)) if len(tr) else float("nan"),
            "decision_action_v0_share": decision_actions.get(0, 0) / dd,
            "decision_action_v1_share": decision_actions.get(1, 0) / dd,
            "episode_reward": total_reward,
            "episode_steps": steps,
            "decision_gate_found": gate_found,
            "final_queue_v0": snapshot["queues"].get("0", 0),
            "final_queue_v1": snapshot["queues"].get("1", 0),
        }
        if decision_probs:
            P = np.vstack(decision_probs)
            H = -np.sum(P * np.log(np.clip(P, 1e-12, 1.0)), axis=1)
            result.update({
                "mean_policy_prob_v0": float(P[:, 0].mean()),
                "mean_policy_prob_v1": float(P[:, 1].mean()),
                "policy_entropy_normalized": float(H.mean() / math.log(2.0)),
                "mean_policy_margin": float(np.abs(P[:, 0] - P[:, 1]).mean()),
            })
        else:
            result.update({
                "mean_policy_prob_v0": float("nan"),
                "mean_policy_prob_v1": float("nan"),
                "policy_entropy_normalized": float("nan"),
                "mean_policy_margin": float("nan"),
            })
        diag = snapshot["diag"] or {}
        result.update({
            "service_arrivals": diag.get("service_arrivals", np.nan),
            "reposition_departures": diag.get("reposition_departures", np.nan),
            "reposition_arrivals": diag.get("reposition_arrivals", np.nan),
            "final_aircraft_states": diag.get("state_counts", {}),
            "final_aircraft_locations": diag.get("current_location_counts", {}),
        })
        return result
    finally:
        env.close()


METRICS = [
    "ATT", "AGT_access", "AWT", "AFT", "completion_rate", "final_backlog",
    "travel_median", "travel_p90", "travel_p95", "travel_max",
    "decision_action_v0_share", "decision_action_v1_share",
    "mean_policy_prob_v0", "mean_policy_prob_v1",
    "policy_entropy_normalized", "mean_policy_margin",
    "final_queue_v0", "final_queue_v1", "service_arrivals",
    "reposition_departures", "reposition_arrivals",
]


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], int(r["train_step"]))].append(r)
    out = []
    for (method, step), g in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        row = {"method": method, "train_step": step, "n_eval_seeds": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = mean(r.get(m) for r in g)
            row[f"{m}_std"] = std(r.get(m) for r in g)
        out.append(row)
    return out


def discover(run_root: Path, final_only: bool):
    specs = []
    for method in METHODS:
        d = run_root / method / "checkpoints"
        if not d.exists():
            raise FileNotFoundError(d)
        one = []
        for p in d.glob("uam_ppo_*_steps.zip"):
            m = re.search(r"uam_ppo_(\d+)_steps", p.name)
            if not m:
                continue
            step = int(m.group(1))
            if step <= 800000 and step % 50000 == 0:
                vec = d / f"uam_ppo_vecnormalize_{step}_steps.pkl"
                if not vec.exists():
                    raise FileNotFoundError(vec)
                one.append((step, p.resolve(), vec.resolve()))
        one.sort()
        if final_only and one:
            one = [one[-1]]
        for x in one:
            specs.append((method,) + x)
    return sorted(specs, key=lambda x: (x[1], x[0]))


def previous_return_home():
    serial = ROOT / "serial_runs"
    runs = [p for p in serial.glob("uagmc_E1_fixed16_fast16env_3seed_1[mM]_*") if p.is_dir()]
    if not runs:
        # glob character class above won't match whole token on Windows Path consistently; explicit fallback
        runs = [p for p in serial.glob("uagmc_E1_fixed16_fast16env_3seed_1M_*") if p.is_dir()]
        runs += [p for p in serial.glob("uagmc_E1_fixed16_fast16env_3seed_1m_*") if p.is_dir()]
    if not runs:
        return None
    root = max(runs, key=lambda p: p.stat().st_mtime)
    d = root / "seed_1" / "checkpoints"
    model = d / "uam_ppo_800000_steps.zip"
    vec = d / "uam_ppo_vecnormalize_800000_steps.pkl"
    if model.exists() and vec.exists():
        return root, model, vec
    return None


def plot_curve(curve, metric, ylabel, path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    for method in METHODS:
        g = [r for r in curve if r["method"] == method]
        if not g:
            continue
        ax.errorbar(
            [int(r["train_step"]) for r in g],
            [ffloat(r.get(f"{metric}_mean")) for r in g],
            yerr=[ffloat(r.get(f"{metric}_std"), 0.0) for r in g],
            marker="o", capsize=3, label=method,
        )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"LQ vs VertiSync-simple: {metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def zip_results(outdir: Path) -> Path:
    zpath = outdir / "UPLOAD_THIS_reposition_1600k_analysis.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob("*")):
            if p.is_file() and p != zpath and p.suffix.lower() in {".csv", ".json", ".txt", ".png"}:
                z.write(p, p.relative_to(outdir).as_posix())
    return zpath


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default=None)
    p.add_argument("--eval-seeds", default="123,124,125")
    p.add_argument("--max-time", type=int, default=600)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--final-only", action="store_true")
    p.add_argument("--skip-return-home-reference", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_root = Path(args.run_root).resolve() if args.run_root else auto_run_root()
    if not PASSENGER_FILE.exists():
        raise FileNotFoundError(PASSENGER_FILE)
    device = "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    eval_seeds = [int(x) for x in args.eval_seeds.split(",") if x.strip()]
    outdir = run_root / "posttrain_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    specs = discover(run_root, args.final_only)
    print("=" * 118)
    print("POST-TRAINING EVALUATION | LQ 800k + VERTISYNC-SIMPLE 800k")
    print("=" * 118)
    print("run root:", run_root)
    print("checkpoints:", len(specs), "eval seeds:", eval_seeds)
    print("=" * 118)

    episodes, errors = [], []
    total = len(specs) * len(eval_seeds)
    job = 0
    for method, step, model, vec in specs:
        for es in eval_seeds:
            job += 1
            print(f"[{job:>3}/{total}] {method:<18} step={step:>7,d} eval_seed={es}", flush=True)
            try:
                r = evaluate_one(method, step, model, vec, es, args.max_time, device, outdir)
                episodes.append(r)
                print(
                    f"    ATT={r['ATT']:.3f} | AWT={r['AWT']:.3f} | "
                    f"finish={r['N_finished']}/{r['N']} | "
                    f"V0={100*r['decision_action_v0_share']:.1f}% | "
                    f"H={r['policy_entropy_normalized']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({
                    "method": method, "train_step": step, "eval_seed": es,
                    "error": repr(exc), "traceback": traceback.format_exc(),
                })
                print("    ERROR:", repr(exc), flush=True)
                if args.fail_fast:
                    raise

    write_csv(outdir / "episode_metrics.csv", episodes)
    curve = aggregate(episodes)
    write_csv(outdir / "curve_by_method.csv", curve)
    final = [r for r in curve if int(r["train_step"]) == 800000]
    write_csv(outdir / "final_800k_comparison.csv", final)

    # Optional previous fixed return-home reference at seed1@800k.
    return_rows = []
    if not args.skip_return_home_reference:
        old = previous_return_home()
        if old is not None:
            oldroot, model, vec = old
            print("\nreturn-home reference:", oldroot)
            for es in eval_seeds:
                try:
                    r = evaluate_one("return_home", 800000, model, vec, es, args.max_time, device, outdir)
                    return_rows.append(r)
                except Exception as exc:
                    errors.append({
                        "method": "return_home", "train_step": 800000,
                        "eval_seed": es, "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    })
    return_summary = aggregate(return_rows) if return_rows else []
    write_csv(outdir / "return_home_reference_800k.csv", return_summary)
    write_csv(outdir / "threeway_final_comparison.csv", final + return_summary)
    write_csv(outdir / "errors.csv", errors)

    # Copy tiny training logs for later analysis.
    for name in ("experiment_manifest.json", "experiment_end.json", "serial_status.csv"):
        src = run_root / name
        if src.exists():
            shutil.copy2(src, outdir / f"training_{name}")
    for method in METHODS:
        for name in ("run_manifest.json", "run_end.json", "training_milestones.csv", "preflight.json"):
            src = run_root / method / name
            if src.exists():
                shutil.copy2(src, outdir / f"{method}_{name}")

    plot_curve(curve, "ATT", "ATT (min)", outdir / "att_curve.png")
    plot_curve(curve, "AWT", "AWT (min)", outdir / "awt_curve.png")
    plot_curve(curve, "completion_rate", "Completion rate", outdir / "completion_curve.png")
    plot_curve(curve, "decision_action_v0_share", "V0 action share", outdir / "action_share_curve.png")

    lines = [
        "UAGMC N=16 reposition post-training evaluation",
        f"run_root={run_root}",
        "",
        "FINAL 800k",
    ]
    for r in final + return_summary:
        lines.append(
            f"{r['method']}: ATT={ffloat(r.get('ATT_mean')):.4f}, "
            f"AWT={ffloat(r.get('AWT_mean')):.4f}, "
            f"completion={100*ffloat(r.get('completion_rate_mean')):.2f}%, "
            f"V0={100*ffloat(r.get('decision_action_v0_share_mean')):.2f}%"
        )
    lines += [
        "",
        "Notes:",
        "- ATT must be interpreted together with completion/backlog.",
        "- 123/124/125 can be identical on the fixed deterministic trace.",
        "- This is one TRAINING seed, suitable for development comparison, not final statistics.",
        "- VertiSync-simple is the exact simplified rule from the training script, not full VertiSync MILP.",
        f"errors={len(errors)}",
    ]
    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    zpath = zip_results(outdir)
    print("\n" + "=" * 118)
    print("DONE")
    print("final:", outdir / "final_800k_comparison.csv")
    print("three-way:", outdir / "threeway_final_comparison.csv")
    print("UPLOAD THIS FILE:")
    print(zpath)
    print("=" * 118)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
