'''
Author: pangay 1623253042@qq.com
Date: 2024-01-11 21:24:41
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2024-01-17 22:12:42
FilePath: /Air_Taxi_simulation/utils/make_env.py
'''
import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from at_obj.scenario import Scenario
from utilss.uam_rl_wrapper import UAMRLWrapper   # 你的 RL wrapper


# ============================
# Env factory (SB3 standard)
# ============================
def make_env(
    max_time: int = 420,
    log_dir: str = "logs",
    env_index: int = 0,
    person_spawn_file=None,
    candidate_from_vertiports=None,
    to_vertiport: int = 2,
    enable_logger: bool = False,
):

    """
    Create a single RL environment instance.

    Args:
        max_time: scenario max simulation time
        log_dir: directory for Monitor logs
        env_index: environment index (for parallel envs)
        person_spawn_file: fixed passenger file for reproducibility
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def _init() -> gym.Env:
        scenario = Scenario(
            max_time=max_time,
            person_spawn_file=person_spawn_file,
            enable_logger = enable_logger,
        )

        env = UAMRLWrapper(
            scenario=scenario,
            candidate_from_vertiports = candidate_from_vertiports,
            to_vertiport = to_vertiport,
        )

        env = Monitor(
            env,
            filename=str(log_dir / f"env_{env_index}.monitor.csv"),
            allow_early_resets=True
        )

        return env


    return _init
