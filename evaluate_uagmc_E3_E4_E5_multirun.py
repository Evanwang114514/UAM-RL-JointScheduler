# -*- coding: utf-8 -*-
"""
Robust multi-run post-training evaluator for UAGMC E3/E4/E5
============================================================

Why this exists
---------------
The original evaluator assumed E3/E4/E5 all lived under ONE serial-run folder.
If training was restarted/resumed stage-by-stage, the stages can live in
different run folders. Then a valid E3/E4 result is incorrectly reported as
"directory missing".

This version scans ALL matching:
    serial_runs/uagmc_E3_E4_E5_LQ_1m_seed*_*

and independently discovers the best completed source for each stage.

Selection priority for each stage
---------------------------------
1) exact formal 1,000,000-step checkpoint + matched VecNormalize
2) successful final_rl_model.zip + final_vec_normalize.pkl
   (only if run_end.json says SUCCESS)

It refuses to silently mix different fleet sizes across stages.

It reuses evaluate_uagmc_E3_E4_E5_auto.py for the actual rollout/metrics and
train_uagmc_E3_E4_E5_serial_1m.py for the exact stage physics.

Run:
    python evaluate_uagmc_E3_E4_E5_multirun.py --final-only

Full learning curves where formal checkpoints exist:
    python evaluate_uagmc_E3_E4_E5_multirun.py

Output:
    serial_runs/E3_E4_E5_MULTIRUN_ANALYSIS_<timestamp>/
        stage_sources.csv
        episode_metrics.csv
        curve_by_stage.csv
        final_comparison.csv
        errors.csv
        summary.txt
        UPLOAD_THIS_E3_E4_E5_MULTIRUN.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import evaluate_uagmc_E3_E4_E5_auto as base
except Exception as exc:
    raise RuntimeError(
        "Cannot import evaluate_uagmc_E3_E4_E5_auto.py. "
        "Put both evaluator files in UAGMC-main."
    ) from exc


STAGES = ("E3_SINGLE_PAX", "E4_TURNAROUND", "E5_PAD")
FINAL_STEP = 1_000_000


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
                    out[k] = json.dumps(base.jsonable(v), ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def discover_run_roots() -> List[Path]:
    serial = ROOT / "serial_runs"

    roots = [
        p.resolve()
        for p in serial.glob("uagmc_E3_E4_E5_LQ_1m_seed*_*")
        if p.is_dir()
    ]

    if not roots:
        roots = [
            p.resolve()
            for p in serial.glob("*E3_E4_E5*LQ*")
            if p.is_dir()
        ]

    roots.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not roots:
        raise FileNotFoundError(
            "No E3/E4/E5 serial run directories found under serial_runs/."
        )

    return roots


def fleet_size_of(root: Path) -> Optional[int]:
    manifest = read_json(root / "experiment_manifest.json")
    value = manifest.get(
        "fleet_size_frozen_for_all_stages",
        manifest.get("fleet_size"),
    )
    try:
        return int(value)
    except Exception:
        return None


def successful_final(stage_dir: Path) -> bool:
    end = read_json(stage_dir / "run_end.json")
    if str(end.get("status", "")).upper() != "SUCCESS":
        return False

    requested = end.get("requested_timesteps")
    try:
        if requested is not None and int(requested) < FINAL_STEP:
            return False
    except Exception:
        pass

    return True


@dataclass(frozen=True)
class StageSource:
    stage: str
    run_root: Path
    fleet_size: int
    source_kind: str
    model: Path
    vec: Path
    step_label: int
    exact_1m: bool


def source_candidates(stage: str, roots: Sequence[Path]) -> List[StageSource]:
    out: List[StageSource] = []

    for root in roots:
        fleet = fleet_size_of(root)
        if fleet is None or fleet <= 0:
            continue

        stage_dir = root / stage
        if not stage_dir.exists():
            continue

        ck = stage_dir / "checkpoints"
        exact_model = ck / f"uam_ppo_{FINAL_STEP}_steps.zip"
        exact_vec = ck / f"uam_ppo_vecnormalize_{FINAL_STEP}_steps.pkl"

        if exact_model.exists() and exact_vec.exists():
            out.append(
                StageSource(
                    stage=stage,
                    run_root=root,
                    fleet_size=fleet,
                    source_kind="EXACT_1M_CHECKPOINT",
                    model=exact_model.resolve(),
                    vec=exact_vec.resolve(),
                    step_label=FINAL_STEP,
                    exact_1m=True,
                )
            )

        final_model = stage_dir / "final_rl_model.zip"
        final_vec = stage_dir / "final_vec_normalize.pkl"

        if (
            final_model.exists()
            and final_vec.exists()
            and successful_final(stage_dir)
        ):
            out.append(
                StageSource(
                    stage=stage,
                    run_root=root,
                    fleet_size=fleet,
                    source_kind="SUCCESSFUL_FINAL_MODEL",
                    model=final_model.resolve(),
                    vec=final_vec.resolve(),
                    step_label=FINAL_STEP,
                    exact_1m=False,
                )
            )

    return out


def choose_sources(roots: Sequence[Path]) -> Tuple[List[StageSource], List[Dict[str, Any]]]:
    chosen: List[StageSource] = []
    notes: List[Dict[str, Any]] = []

    # Prefer the most common/most recent viable fleet size, but never silently mix.
    all_by_stage = {stage: source_candidates(stage, roots) for stage in STAGES}

    fleet_support: Dict[int, int] = {}
    for stage, candidates in all_by_stage.items():
        for fleet in {c.fleet_size for c in candidates}:
            fleet_support[fleet] = fleet_support.get(fleet, 0) + 1

    if not fleet_support:
        return [], [
            {
                "stage": stage,
                "status": "MISSING",
                "reason": "no exact 1M checkpoint or successful final model found",
            }
            for stage in STAGES
        ]

    # Max number of stages supported; tie -> newest source among that fleet.
    best_fleet = max(
        fleet_support,
        key=lambda f: (
            fleet_support[f],
            max(
                (
                    c.run_root.stat().st_mtime
                    for cs in all_by_stage.values()
                    for c in cs
                    if c.fleet_size == f
                ),
                default=0.0,
            ),
        ),
    )

    for stage in STAGES:
        candidates = [
            c for c in all_by_stage[stage]
            if c.fleet_size == best_fleet
        ]

        if not candidates:
            notes.append(
                {
                    "stage": stage,
                    "status": "MISSING",
                    "fleet_size": best_fleet,
                    "reason": (
                        "no completed source found at the common fleet size "
                        f"N={best_fleet}"
                    ),
                }
            )
            continue

        # Exact 1M first; within same kind take newest run.
        candidates.sort(
            key=lambda c: (
                1 if c.exact_1m else 0,
                c.run_root.stat().st_mtime,
            ),
            reverse=True,
        )
        pick = candidates[0]
        chosen.append(pick)

        notes.append(
            {
                "stage": stage,
                "status": "SELECTED",
                "fleet_size": pick.fleet_size,
                "source_kind": pick.source_kind,
                "run_root": str(pick.run_root),
                "model": str(pick.model),
                "vecnormalize": str(pick.vec),
                "exact_1m": pick.exact_1m,
            }
        )

    return chosen, notes


def formal_curve_specs(source: StageSource, eval_every: int) -> List[base.Spec]:
    """
    For a selected stage source, use formal checkpoints from the same run root.
    If only final model exists, include that final model as 1M.
    """
    stage_dir = source.run_root / source.stage
    ck = stage_dir / "checkpoints"

    specs: List[base.Spec] = []

    if ck.exists():
        for model in ck.glob("uam_ppo_*_steps.zip"):
            m = re.search(r"uam_ppo_(\d+)_steps", model.name)
            if not m:
                continue

            step = int(m.group(1))
            if step > FINAL_STEP or step % eval_every != 0:
                continue

            vec = ck / f"uam_ppo_vecnormalize_{step}_steps.pkl"
            if vec.exists():
                specs.append(
                    base.Spec(
                        source.stage,
                        step,
                        model.resolve(),
                        vec.resolve(),
                    )
                )

    specs.sort(key=lambda s: s.step)

    # Ensure a final point exists even if CheckpointCallback missed exact 1M.
    if not any(s.step == FINAL_STEP for s in specs):
        specs.append(
            base.Spec(
                source.stage,
                FINAL_STEP,
                source.model,
                source.vec,
            )
        )

    return specs


def build_zip(outdir: Path) -> Path:
    zpath = outdir / "UPLOAD_THIS_E3_E4_E5_MULTIRUN.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob("*")):
            if not p.is_file() or p == zpath:
                continue
            if p.suffix.lower() in {".csv", ".json", ".txt"}:
                z.write(p, p.relative_to(outdir).as_posix())
    return zpath


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-seeds", default="123,124,125")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--max-time", type=int, default=600)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--final-only", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    roots = discover_run_roots()
    chosen, source_rows = choose_sources(roots)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = (
        ROOT
        / "serial_runs"
        / f"E3_E4_E5_MULTIRUN_ANALYSIS_{stamp}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    monitor = outdir / "_monitor"
    monitor.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "stage_sources.csv", source_rows)

    print("=" * 126)
    print("UAGMC E3/E4/E5 MULTI-RUN AUTO DISCOVERY")
    print("=" * 126)
    print(f"Run folders scanned: {len(roots)}")
    for row in source_rows:
        if row["status"] == "SELECTED":
            print(
                f"{row['stage']:<20} | SELECTED | N={row['fleet_size']} | "
                f"{row['source_kind']} | {Path(row['run_root']).name}"
            )
        else:
            print(
                f"{row['stage']:<20} | MISSING  | {row.get('reason','')}"
            )
    print("=" * 126)

    if not chosen:
        print("No completed stage source found.")
        return 0

    # Common fleet has already been enforced by choose_sources().
    fleet_size = chosen[0].fleet_size

    if len({c.fleet_size for c in chosen}) != 1:
        raise RuntimeError("Internal error: selected stages have different fleet sizes.")

    eval_seeds = base.parse_ints(args.eval_seeds)
    device = "cpu" if args.device == "auto" else args.device

    specs: List[base.Spec] = []
    for source in chosen:
        if args.final_only:
            specs.append(
                base.Spec(
                    source.stage,
                    FINAL_STEP,
                    source.model,
                    source.vec,
                )
            )
        else:
            specs.extend(
                formal_curve_specs(
                    source,
                    int(args.eval_every),
                )
            )

    # Remove duplicate stage/step specs, preferring exact formal checkpoint paths.
    dedup: Dict[Tuple[str, int], base.Spec] = {}
    for s in specs:
        dedup[(s.stage, s.step)] = s
    specs = sorted(
        dedup.values(),
        key=lambda s: (STAGES.index(s.stage), s.step),
    )

    write_csv(
        outdir / "checkpoint_inventory.csv",
        [
            {
                "stage": s.stage,
                "train_step": s.step,
                "model_path": str(s.model),
                "vecnormalize_path": str(s.vec),
            }
            for s in specs
        ],
    )

    rows, errors = [], []
    total = len(specs) * len(eval_seeds)
    job = 0

    for spec in specs:
        for eval_seed in eval_seeds:
            job += 1
            print(
                f"[{job:>3}/{total}] {spec.stage:<20} "
                f"step={spec.step:>9,d} eval_seed={eval_seed}",
                flush=True,
            )

            try:
                row = base.run_episode(
                    spec,
                    fleet_size,
                    eval_seed,
                    int(args.max_time),
                    device,
                    monitor,
                )
                rows.append(row)

                print(
                    f"    ATT={row['ATT']:.3f} | "
                    f"AWT={row['AWT']:.3f} | "
                    f"finish={row['N_finished']}/{row['N']} | "
                    f"V0={100*row['decision_action_v0_share']:.1f}% | "
                    f"H={row['policy_entropy_normalized']:.3f}",
                    flush=True,
                )

            except Exception as exc:
                err = {
                    "stage": spec.stage,
                    "train_step": spec.step,
                    "eval_seed": eval_seed,
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

    curve = base.aggregate(rows)
    write_csv(outdir / "curve_by_stage.csv", curve)

    final_rows = [
        r for r in curve
        if int(r["train_step"]) == FINAL_STEP
    ]
    write_csv(outdir / "final_comparison.csv", final_rows)

    lines = [
        "=" * 126,
        "UAGMC E3/E4/E5 MULTI-RUN POST-TRAINING ANALYSIS",
        "=" * 126,
        f"Common fleet size: {fleet_size}",
        "",
        "SELECTED SOURCES",
        "-" * 126,
    ]

    for row in source_rows:
        if row["status"] == "SELECTED":
            lines.append(
                f"{row['stage']:<20} | {row['source_kind']:<24} | "
                f"{Path(row['run_root']).name}"
            )
        else:
            lines.append(
                f"{row['stage']:<20} | MISSING | {row.get('reason','')}"
            )

    lines += ["", "FINAL COMPARISON", "-" * 126]

    for r in final_rows:
        lines.append(
            f"{r['stage']:<20} | "
            f"ATT={base.fnum(r.get('ATT_mean')):.3f} | "
            f"AWT={base.fnum(r.get('AWT_mean')):.3f} | "
            f"finish={100*base.fnum(r.get('completion_rate_mean')):.2f}% | "
            f"V0={100*base.fnum(r.get('decision_action_v0_share_mean')):.1f}% | "
            f"H={base.fnum(r.get('policy_entropy_normalized_mean')):.3f}"
        )

    lines += [
        "",
        "NOTES",
        "-" * 126,
        "1) Stages may come from different serial-run folders, but only a common fleet size is allowed.",
        "2) Exact 1M checkpoint is preferred. Successful final model is fallback only.",
        "3) ATT must be read together with completion/backlog.",
        "4) This is still one training seed per stage unless your source folders contain other seeds.",
        f"Errors: {len(errors)}",
    ]

    (outdir / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    write_json(
        outdir / "analysis_manifest.json",
        {
            "common_fleet_size": fleet_size,
            "selected_sources": source_rows,
            "eval_seeds": eval_seeds,
            "eval_every": int(args.eval_every),
            "max_time": int(args.max_time),
            "final_only": bool(args.final_only),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    bundle = build_zip(outdir)

    print("\n" + "=" * 126)
    print("MULTI-RUN ANALYSIS COMPLETE")
    print("=" * 126)
    print(f"Sources : {outdir / 'stage_sources.csv'}")
    print(f"Curve   : {outdir / 'curve_by_stage.csv'}")
    print(f"Final   : {outdir / 'final_comparison.csv'}")
    print(f"Summary : {outdir / 'summary.txt'}")
    print("-" * 126)
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(bundle)
    print("=" * 126)

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
