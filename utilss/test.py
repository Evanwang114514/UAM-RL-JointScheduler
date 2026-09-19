'''
Author: pangay 1623253042@qq.com
Date: 2024-01-14 21:19:12
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2024-01-22 13:54:57
'''
import sys
from pathlib import Path

parent_directory = Path(__file__).resolve().parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))
    
    
import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from gym import logger, spaces
from stable_baselines3.common.env_checker import check_env

from utilss.make_env import make_env


if __name__ == '__main__':

    at_env_generate = make_env(time=100, log_file = 'test')
    tsc_env = at_env_generate()

    # Check Env
    print(tsc_env.observation_space.sample())
    check_env(tsc_env)
    # Simulation with environment
    dones = False
    tsc_env.reset()
    dones = False
    rewards = 0
    while not dones:
        action = [1,1,1,0]
        states, reward, truncated, dones, infos = tsc_env.step(action=action)
        rewards += reward
        #logger.info(f"SIM: {infos['step_time']} \n+State:{states}; \n+Reward:{rewards}.")
    tsc_env.close()

    print('rewarde',rewards)
