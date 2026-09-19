# -*- coding: utf-8 -*-
r"""
Fixed-path learning-curve evaluator for UAGMC E3/E4/E5
======================================================

This script DOES NOT auto-discover the newest run.
It evaluates the known completed experiment root directly:

E:\Study Files\github\UAM-predict\UAGMC-main\serial_runs\
uagmc_E3_E4_E5_LQ_1m_seed1_20260919_003109

Stages:
    E3_SINGLE_PAX
    E4_TURNAROUND
    E5_PAD

It scans formal checkpoints:
    50k, 100k, ..., 1,000k

Default eval seeds:
    123,124,125

It reuses rollout/metrics from:
    evaluate_uagmc_E3_E4_E5_auto.py

Run:
    python evaluate_uagmc_E3_E4_E5_curves_fixedpath.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import evaluate_uagmc_E3_E4_E5_auto as base
except Exception as exc:
    raise RuntimeError(
        "Cannot import evaluate_uagmc_E3_E4_E5_auto.py. "
        "Put this file beside it in UAGMC-main."
    ) from exc


DEFAULT_RUN_ROOT = Path(
    r"E:\Study Files\github\UAM-predict\UAGMC-main\serial_runs\uagmc_E3_E4_E5_LQ_1m_seed1_20260919_003109"
)

STAGES = (
    "E3_SINGLE_PAX",
    "E4_TURNAROUND",
    "E5_PAD",
)

DEFAULT_STEPS = list(range(50_000, 1_000_001, 50_000))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for row in rows:
            out = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    out[key] = json.dumps(
                        base.jsonable(value),
                        ensure_ascii=False,
                    )
                else:
                    out[key] = value
            w.writerow(out)


def parse_ints(text: str) -> List[int]:
    values = [
        int(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]
    if not values:
        raise ValueError("empty integer list")
    return values


def read_manifest(run_root: Path) -> Dict[str, Any]:
    path = run_root / "experiment_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def resolve_fleet_size(manifest: Dict[str, Any]) -> int:
    fleet_size = int(
        manifest.get(
            "fleet_size_frozen_for_all_stages",
            manifest.get("fleet_size", -1),
        )
    )
    if fleet_size <= 0:
        raise RuntimeError(
            "Cannot resolve frozen fleet size from experiment_manifest.json"
        )
    return fleet_size


def build_checkpoint_specs(
    run_root: Path,
    requested_steps: Sequence[int],
):
    specs = []
    inventory = []

    for stage in STAGES:
        checkpoint_dir = run_root / stage / "checkpoints"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Missing checkpoint directory for {stage}: {checkpoint_dir}"
            )

        found = 0

        for step in requested_steps:
            model = checkpoint_dir / f"uam_ppo_{step}_steps.zip"
            vec = checkpoint_dir / f"uam_ppo_vecnormalize_{step}_steps.pkl"

            ok = model.exists() and vec.exists()

            inventory.append(
                {
                    "stage": stage,
                    "train_step": step,
                    "model_exists": model.exists(),
                    "vecnormalize_exists": vec.exists(),
                    "selected_for_eval": ok,
                    "model_path": str(model),
                    "vecnormalize_path": str(vec),
                }
            )

            if ok:
                specs.append(
                    base.Spec(
                        stage=stage,
                        step=int(step),
                        model=model.resolve(),
                        vec=vec.resolve(),
                    )
                )
                found += 1

        if found == 0:
            raise RuntimeError(
                f"No matched model+VecNormalize checkpoint found for {stage}"
            )

    specs.sort(
        key=lambda x: (
            STAGES.index(x.stage),
            x.step,
        )
    )

    return specs, inventory


def make_plot(curve, metric: str, ylabel: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_subplot(111)

    for stage in STAGES:
        rows = [r for r in curve if r["stage"] == stage]
        if not rows:
            continue

        x = [int(r["train_step"]) for r in rows]
        y = [base.fnum(r.get(f"{metric}_mean")) for r in rows]
        err = [base.fnum(r.get(f"{metric}_std"), 0.0) for r in rows]

        ax.errorbar(
            x,
            y,
            yerr=err,
            marker="o",
            capsize=3,
            label=stage,
        )

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"E3 / E4 / E5 learning curves: {metric}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_zip(output_dir: Path) -> Path:
    zpath = output_dir / "UPLOAD_THIS_E3_E4_E5_CURVES.zip"

    with zipfile.ZipFile(
        zpath,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            if path == zpath:
                continue
            if path.suffix.lower() not in {
                ".csv",
                ".json",
                ".txt",
                ".png",
            }:
                continue

            z.write(
                path,
                path.relative_to(output_dir).as_posix(),
            )

    return zpath


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate fixed known E3/E4/E5 training curves."
    )

    p.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
    )
    p.add_argument(
        "--eval-seeds",
        default="123,124,125",
    )
    p.add_argument(
        "--max-time",
        type=int,
        default=600,
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    p.add_argument(
        "--steps",
        default=",".join(str(x) for x in DEFAULT_STEPS),
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    run_root = Path(args.run_root).expanduser()
    if not run_root.is_absolute():
        run_root = (ROOT / run_root).resolve()
    else:
        run_root = run_root.resolve()

    if not run_root.exists():
        raise FileNotFoundError(
            f"Run root does not exist: {run_root}"
        )

    manifest = read_manifest(run_root)
    fleet_size = resolve_fleet_size(manifest)

    eval_seeds = parse_ints(args.eval_seeds)
    steps = parse_ints(args.steps)

    device = "cpu" if args.device == "auto" else args.device

    specs, inventory = build_checkpoint_specs(
        run_root=run_root,
        requested_steps=steps,
    )

    output_dir = run_root / "curve_eval_fixedpath"
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_dir = output_dir / "_monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "checkpoint_inventory.csv",
        inventory,
    )

    print("=" * 124)
    print("UAGMC E3 / E4 / E5 FIXED-PATH CURVE EVALUATION")
    print("=" * 124)
    print(f"Run root    : {run_root}")
    print(f"Fleet size  : {fleet_size}")
    print(f"Stages      : {STAGES}")
    print(f"Checkpoints : {len(specs)} matched pairs")
    print(f"Eval seeds  : {eval_seeds}")
    print(f"Device      : {device}")
    print("=" * 124)

    rows = []
    errors = []

    total_jobs = len(specs) * len(eval_seeds)
    job = 0

    for spec in specs:
        for eval_seed in eval_seeds:
            job += 1

            print(
                f"[{job:>3}/{total_jobs}] "
                f"{spec.stage:<20} "
                f"step={spec.step:>9,d} "
                f"eval_seed={eval_seed}",
                flush=True,
            )

            try:
                row = base.run_episode(
                    spec,
                    fleet_size,
                    eval_seed,
                    int(args.max_time),
                    device,
                    monitor_dir,
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
                    "model_path": str(spec.model),
                    "vecnormalize_path": str(spec.vec),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)

                print(
                    f"    ERROR: {repr(exc)}",
                    flush=True,
                )

                if args.fail_fast:
                    write_csv(
                        output_dir / "errors.csv",
                        errors,
                    )
                    raise

    write_csv(
        output_dir / "episode_metrics.csv",
        rows,
    )
    write_csv(
        output_dir / "errors.csv",
        errors,
    )

    curve = base.aggregate(rows)

    write_csv(
        output_dir / "curve_by_stage.csv",
        curve,
    )

    final_rows = [
        r for r in curve
        if int(r["train_step"]) == 1_000_000
    ]

    write_csv(
        output_dir / "final_1m_comparison.csv",
        final_rows,
    )

    summary_lines = [
        "=" * 124,
        "UAGMC E3 / E4 / E5 FIXED-PATH LEARNING CURVES",
        "=" * 124,
        f"Run root  : {run_root}",
        f"Fleet size: {fleet_size}",
        "",
        "FINAL 1M",
        "-" * 124,
    ]

    for r in final_rows:
        summary_lines.append(
            f"{r['stage']:<20} | "
            f"ATT={base.fnum(r.get('ATT_mean')):8.3f} | "
            f"AWT={base.fnum(r.get('AWT_mean')):8.3f} | "
            f"finish={100*base.fnum(r.get('completion_rate_mean')):6.2f}% | "
            f"V0={100*base.fnum(r.get('decision_action_v0_share_mean')):6.2f}% | "
            f"H={base.fnum(r.get('policy_entropy_normalized_mean')):6.3f}"
        )

    summary_lines += [
        "",
        "NOTES",
        "-" * 124,
        "This evaluator uses the fixed 20260919_003109 model root by default.",
        "No newest-run auto-discovery is used.",
        "Missing intermediate checkpoints are skipped.",
        "ATT must be read together with completion/backlog, especially for E5.",
        f"Errors: {len(errors)}",
    ]

    (output_dir / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    make_plot(
        curve,
        "ATT",
        "ATT (min)",
        output_dir / "ATT_curve.png",
    )
    make_plot(
        curve,
        "AWT",
        "AWT (min)",
        output_dir / "AWT_curve.png",
    )
    make_plot(
        curve,
        "completion_rate",
        "Completion rate",
        output_dir / "completion_curve.png",
    )
    make_plot(
        curve,
        "decision_action_v0_share",
        "V0 deterministic action share",
        output_dir / "V0_share_curve.png",
    )
    make_plot(
        curve,
        "policy_entropy_normalized",
        "Normalized policy entropy",
        output_dir / "entropy_curve.png",
    )

    shutil.copy2(
        run_root / "experiment_manifest.json",
        output_dir / "training_experiment_manifest.json",
    )

    bundle = build_zip(output_dir)

    print("\n" + "=" * 124)
    print("FIXED-PATH CURVE EVALUATION COMPLETE")
    print("=" * 124)
    print(f"Curve CSV : {output_dir / 'curve_by_stage.csv'}")
    print(f"Final 1M  : {output_dir / 'final_1m_comparison.csv'}")
    print(f"Summary   : {output_dir / 'summary.txt'}")
    print("-" * 124)
    print("UPLOAD THIS FILE TO CHATGPT:")
    print(bundle)
    print("=" * 124)

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
