'''
Author: pangay 1623253042@qq.com
Date: 2024-01-11 21:21:42
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2025-05-13 14:57:43
FilePath: /Air_Taxi_simulation/utils/at_wrapper.py
'''
import numpy as np
import gymnasium as gym
import csv

#from gym import spaces
from gymnasium import spaces  #考虑不同版本
from gymnasium.core import Env
from gym.spaces import Box
from collections import deque
from typing import Any, SupportsFloat, Tuple, Dict

from loguru import logger

# state  乘客 起点，终点
# 前往不同 vertiport 的人数
# 每个 vertiport 等待的人数
# vertiport 中每个 eVTOL 的状态
# eVTOL所在的 vertiport 载客人数，飞行速度， 充电时间
# action  [0,1]
# 特征提取函数 encoder/decoder 进行决策
# 软件的自成长/更新性质/文件更改后，主动重新加载，进行局部更新/是否可行？

class atWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction with tls_id
    """
    def __init__(self, env: Env) -> None:
        super().__init__(env)


        # Dynamic Information
        self.state = None # 当前的 state
        max_states = 5 #选取的帧数
        self.states = deque([self._get_initial_state()] * max_states, maxlen=max_states) 
        self.action_space =  spaces.MultiBinary(4)
        self.observation_space = spaces.Box(low= 0, high = 1000, shape=(max_states, 9, 4), dtype = np.int64)
        self.past_reward = 0
        # 4 个行人的位置 起点 终点 
        # 机场的位置 两个坐标 
        # 车的速度 飞机的速度 排队等待的人数  
    
    def _get_initial_state(self):
        # 返回初始状态，这里假设所有状态都为 0
        return np.zeros((9,4))
    
    def get_state(self):
        """将 state 从二维 (5, 12) 转换为一维 (1, 60)
        """
        new_state = dict()
        
        new_state = np.array(
                self.states, 
                dtype=np.int64
            ).reshape((-1,9,4))
        return new_state

    def state_wrapper(self, state, vehicle2person, info_wrapper):
        """构建state的格式
        """
        state_wrapper = []
        for i in state['people']: #get passenger position 
              destination = state['people'][i].get_state()
              state_wrapper.append(destination[i][0])
              state_wrapper.append(destination[i][1])
        state_wrapper = np.array(state_wrapper)
        state_wrapper = state_wrapper.reshape(4,-1) #passenger position
        uam_info_list = []
        for i in range(0,2):
            uam_info = state['UAM'][i].get_state()
        #静态信息
            uam_positon = np.array([0,0,0,0])
            uam_positon[0:2] = uam_info[0]
            uam_positon[2:] = uam_info[1]
            uam_info_list.append(uam_positon)
        #动态信息
            uam_wait = np.array([0,0,0,0])
            uam_wait[0] = uam_info[2] # wait person
            uam_wait[1] = uam_info[3] # wait time
            uam_wait[2] = uam_info[4] # speed
            uam_wait[3] = uam_info[5] # fly time
            uam_info_list.append(uam_wait)
        uam_info_list = np.array(uam_info_list)
        #state_wrapper = np.insert(state_wrapper, 0, values=uam_positon, axis=0
        state_wrapper = np.vstack((state_wrapper,uam_info_list))
        state_wrapper = np.insert(state_wrapper, 8, values=info_wrapper, axis=0)

        return state_wrapper
    
    # 机动车匹配模型 #重新写 match 不需要提前去接，直接默认接到
    def vehicle_match(self, state):
        vehicle2person = []
        for i in state['people']: 
            distance_list = []
            vehicle_list = []
            for j in state['vehicle']:
                distance = state['vehicle'][j].get_drive_time(
                      state['vehicle'][j].origin_position,
                      state['people'][i].origin_position
                      )
                distance_list.append(distance)
                vehicle_list.append(j)
             #加入容错机制 防止一辆车匹配到两个人
            temp = sorted(range(len(distance_list)),key=lambda k: distance_list[k])
            vehicle_id = temp[0]
            count = vehicle2person.count(vehicle_id)
            k = 0
            while count != 0:
                k += 1
                vehicle_id = temp[k]
                count = vehicle2person.count(vehicle_id)
        
            vehicle2person.append(vehicle_id)

        return vehicle2person

    # action = 0,1
    def action_wrapper(self, action, vehicle2person):
        action_wrapper = {}
        i = 0 
        for people in self.state['people']:
            #[vehicle_id, if choose UAM] it can chouse 'UAM' or 'ground'
            action_wrapper[people] = [vehicle2person[i], action[i]]
            i += 1

        return action_wrapper

    def info_wrapper(self,info):
        temp_state = [0,0,0,0]
        for nam in info['persons'].person:
            if info['persons'].person[nam].state == 'v':
                vertiport_up_position =int(info['persons'].person[nam].vertiport_up_position)
                temp_state[vertiport_up_position] += 1

        return temp_state

    def reward_wrapper(self, reward):

        return -1 * reward
    
    def reset(self, seed=0) -> Tuple[Any, Dict[str, Any]]:
        
        state =  self.env.reset()
        self.past_reward = 0
        self.state = state
        self.vehicle2person = self.vehicle_match(state)
        info_wrapper = [0,0,0,0]
        state_wrapper = self.state_wrapper(state = state, vehicle2person = self.vehicle2person, info_wrapper = info_wrapper )
        self.states.append(state_wrapper)
        state_wrapper = self.get_state()
        info =  {'uam_fly_time':0, 'uam_wait_time':0, 'reward_drive_time':0,'reward_match_time':0}
        return state_wrapper, info
    

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """更新路口的 state
        """
        action = self.action_wrapper(action = action, vehicle2person = self.vehicle2person)
        state, rewards, truncated, dones, info = super().step(action) # 与环境交互
        if dones == True:
            with open('{}.csv'.format('rl_encode'), 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, dialect='excel')
                person_state = []
                for nam in info['persons'].person:
                    vertiport_up_position = info['persons'].person[nam].vertiport_up_position
                    temp = info['persons'].person[nam].state_list
                    temp.append(int(info['persons'].person[nam].vertiport_up_position))
                    writer.writerow(temp)
                    #logger.info(f"SIM: {info['persons'].person[nam].vertiport_up_position} \n + {info['persons'].person[nam].state_list}")
                    person_state.append(temp)
        self.state = state
        self.vehicle2person = self.vehicle_match(state)
        info_wrapper = self.info_wrapper (info = info)
        state_wrapper = self.state_wrapper(state = state, vehicle2person = self.vehicle2person, info_wrapper = info_wrapper) # 处理每一帧的数据
        self.states.append(state_wrapper)
        reward_wrapper = self.reward_wrapper(rewards)
        state_now = self.get_state()
        return state_now, reward_wrapper, truncated, dones, info
    

    def close(self) -> None:
        return super().close()
    