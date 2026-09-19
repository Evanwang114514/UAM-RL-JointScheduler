'''
Author: pangay 1623253042@qq.com
Date: 2024-01-16 20:38:03
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2024-01-21 23:01:25
'''
#%%
import sys
from pathlib import Path

import numpy as np
from gym import logger, spaces
from stable_baselines3.common.env_checker import check_env

from utilss.make_env import make_env

#%%
def greedy_policy(state):
    action = [0,0,0,0]
    persons_station = state[0:4]
    for i in range(0,4):
        person_station = persons_station[i,:]
        station_0 = get_time(person_station,[0,0],state[5,1])
        station_1 = get_time(person_station,[10,10],state[7,1])
        if station_1 < station_0:
            action[i] = 1
    return action

def get_time(person_station, vertiport_stateion, wait_time):

    travel_time=(
            abs(person_station[0]-vertiport_stateion[0]) 
            + abs(person_station[1]-vertiport_stateion[1])
            )/1.5 # manhattan distance
    total_time = travel_time + wait_time

    return total_time

def get_uam_time(state):
    uam_time = state[1] + state[2]
    return uam_time

at_env_generate = make_env(log_file = 'greedy.log')
tsc_env = at_env_generate()
states, _ = tsc_env.reset()
states = states[-1,:,:]
dones = False
rewards = 0
while not dones:
    action = greedy_policy(states)
    states, reward, truncated, dones, infos = tsc_env.step(action=action)
    states = states[-1,:,:]
tsc_env.close()

print('平均等待时间为',reward)