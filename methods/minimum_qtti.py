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
def queue_policy(state,infos):
    action = [0,0,0,0]
    info = info_wrapper(infos)
    persons_station = state[0:4]
    for i in range(0,4):
        person_station = persons_station[i,:]
        station_0 = get_time(person_station,[0,0],state[5,1],info[0]/(2.5*12))
        station_1 = get_time(person_station,[10,10],state[7,1],info[1]/(1.5*4))
        if station_1 < station_0:
            action[i] = 1
    return action

def get_time(person_station, vertiport_stateion, wait_time, E_lamda):

    travel_time=(
            abs(person_station[0]-vertiport_stateion[0]) 
            + abs(person_station[1]-vertiport_stateion[1])
            )/1.5 # manhattan distance
    E_wait_time = travel_time * E_lamda
    print(travel_time,wait_time,E_wait_time)
    total_time = travel_time + wait_time + E_wait_time

    return total_time

def get_uam_time(state):
    uam_time = state[1] + state[2]
    return uam_time

def info_wrapper(info):
    temp_state = [0,0,0,0]
    for nam in info['persons'].person:
        if info['persons'].person[nam].state == 'v':
            vertiport_up_position =int(info['persons'].person[nam].vertiport_up_position)
            temp_state[vertiport_up_position] += 1

    return temp_state
at_env_generate = make_env(log_file = 'greedy.log')
tsc_env = at_env_generate()
states, _ = tsc_env.reset()
states = states[-1,:,:]
dones = False
rewards = 0
action = [1,1,1,1]
while not dones:
    states, reward, truncated, dones, infos = tsc_env.step(action = action)
    states = states[-1,:,:]
    action = queue_policy(states, infos)
tsc_env.close()

print('平均等待时间为',reward)