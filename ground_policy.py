'''
Author: pangay 1623253042@qq.com
Date: 2024-01-16 20:38:03
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2024-01-21 23:01:25
'''

import sys
from pathlib import Path

import numpy as np
from gym import logger, spaces
from stable_baselines3.common.env_checker import check_env

from utilss.make_env import make_env


def greedy_policy(state):
    action = [0,0,0,0]

    return action

def ground_policy(state):
    persons_station = state[0:4]
    temp = 0
    for i in range(0,4):
        person_station = persons_station[i,:]
        station_0 = get_time(person_station)
        temp +=  station_0
    return temp


def get_time(person_station):

    travel_time=(
            abs(person_station[0]-person_station[2]) 
            + abs(person_station[1]-person_station[3])
            )/2 # manhattan distance
    total_time = travel_time

    return travel_time

def get_uam_time(state):
    uam_time = state[1] + state[2]
    return uam_time

at_env_generate = make_env(log_file = 'greedy.log')
tsc_env = at_env_generate()
states, _ = tsc_env.reset()
dones = False
rewards = 0
while not dones:
    rewards += ground_policy(states)
    action = greedy_policy(states)
    states, reward, truncated, dones, infos = tsc_env.step(action=action)
tsc_env.close()

print('平均等待时间为',rewards/1200)