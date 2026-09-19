'''
Author: pangay 1623253042@qq.com
Date: 2024-01-14 21:19:12
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2024-01-22 12:39:38
'''
import sys
from pathlib import Path

parent_directory = Path(__file__).resolve().parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))
    
import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from stable_baselines3.common.env_checker import check_env

from make_env import make_env


if __name__ == '__main__':

    at_env_generate = make_env(time=100)
    tsc_env = at_env_generate()
    
    # Check Env
    print(tsc_env.observation_space.sample())
    check_env(tsc_env)
    # Simulation with environment
    tsc_env.reset(seed=0)
    dones = False
    rewards = 0
    while not dones:
        action = [1,0,0,0] #action 对于站点的选择
        states, reward, truncated, dones, infos = tsc_env.step(action=action)
    print('rewards',reward)
    tsc_env.close()