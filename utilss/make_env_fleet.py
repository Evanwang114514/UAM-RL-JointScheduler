# -*- coding: utf-8 -*-
"""
Drop-in UAGMC environment factory with a fleet-mode switch.

fleet_mode="legacy_replenish"
    -> official Scenario, unchanged.

fleet_mode="conserved_closed_loop"
    -> fixed aircraft ID set + automatic return-to-home reposition.

This file does NOT replace utilss/make_env.py. Put it beside it as:
    utilss/make_env_fleet.py
"""
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from at_obj.scenario import Scenario
from at_obj.scenario_fixed_fleet import ConservedFleetScenario
from utilss.uam_rl_wrapper import UAMRLWrapper


VALID_FLEET_MODES = {
    "legacy_replenish",
    "conserved_closed_loop",
}


def make_env(
    max_time: int = 10000,
    log_dir: str = "logs",
    env_index: int = 0,
    person_spawn_file=None,
    candidate_from_vertiports=None,
    to_vertiport: int = 2,
    enable_logger: bool = False,
    fleet_mode: str = "legacy_replenish",
    fleet_size: Optional[int] = None,
    fleet_assertions: bool = True,
):
    if fleet_mode not in VALID_FLEET_MODES:
        raise ValueError(
            f"fleet_mode={fleet_mode!r}; "
            f"valid={sorted(VALID_FLEET_MODES)}"
        )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def _init() -> gym.Env:
        if fleet_mode == "legacy_replenish":
            # OFF means literally the public UAGMC Scenario.
            scenario = Scenario(
                max_time=max_time,
                person_spawn_file=person_spawn_file,
                enable_logger=enable_logger,
            )
        else:
            if candidate_from_vertiports is None:
                raise ValueError(
                    "conserved_closed_loop requires "
                    "candidate_from_vertiports, e.g. [0, 1]"
                )

            scenario = ConservedFleetScenario(
                max_time=max_time,
                person_spawn_file=person_spawn_file,
                enable_logger=enable_logger,
                fleet_size=fleet_size,
                fleet_departure_vertiports=candidate_from_vertiports,
                fleet_return_vertiport=to_vertiport,
                fleet_assertions=fleet_assertions,
            )

        env = UAMRLWrapper(
            scenario=scenario,
            candidate_from_vertiports=candidate_from_vertiports,
            to_vertiport=to_vertiport,
        )

        return Monitor(
            env,
            filename=str(log_dir / f"env_{env_index}.monitor.csv"),
            allow_early_resets=True,
        )

    return _init
