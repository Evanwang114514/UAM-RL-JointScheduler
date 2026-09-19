# -*- coding: utf-8 -*-
"""
E0 + E2 T3 supplementary UAGMC training, 700k each
===================================================

Why this file exists
--------------------
E3/E4/E5/E6 T3 UAGMC-style reproductions already exist in the completed
observation-anatomy matrix.  This file only fills the missing T3 source-style
UAGMC controls for:

E0: source UAGMC legacy automatic replenishment.
E2: conserved fixed fleet + responsive Longest-Queue reposition, while source
    UAGMC service batch/capacity remains unchanged.

Everything on the learning side remains UAGMC-style:
    source observation semantics + 6-frame history + existing TemporalLSTM/
    STIN path + PPO/MCSE code already used by the validated project trainer.

Compute profile (yesterday's measured winner)
---------------------------------------------
16 SubprocVecEnv CPU simulators x 1280 steps = 20,480 global rollout
batch_size = 2,048, n_epochs = 10
PPO update on CUDA; deterministic post-training checkpoint evaluation on CPU.

Default training: 700,000 timesteps per stage, train seed=1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import train_uagmc_E0_E2_E6_effect_time_700k as core


def parse_args():
    p = argparse.ArgumentParser(description="E0+E2 T3 UAGMC supplementary training, 700k each")
    p.add_argument("--timesteps", type=int, default=700_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--output-root", default=None)
    p.add_argument("--eval-seeds", default="123,124,125")
    p.add_argument("--analysis-steps", default="200000,400000,500000,550000,600000,650000,700000")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eval_seeds = core.parse_ints(args.eval_seeds)
    analysis_steps = core.parse_ints(args.analysis_steps)

    core.run_experiment(
        stages=["E0", "E2"],
        topology="T3",
        encoder_mode="uagmc",
        timesteps=int(args.timesteps),
        seed=int(args.seed),
        device=str(args.device),
        output_root=(str(Path(args.output_root).expanduser()) if args.output_root else None),
        analysis_steps=analysis_steps,
        eval_seeds=eval_seeds,
        continue_on_error=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
