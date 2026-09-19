'''
Author: PangAY
Date: 2023-12-08 22:45:18
LastEditTime: 2024-01-22 13:15:49
LastEditors: pangay 1623253042@qq.com
'''
import numpy as np
from typing import List
class Map(object):
    
    def __init__(self, 
                 map_size: List[int] = [60,60],
                 map_imformaton: int = [[]*100,[]*100] 
                 ) -> None:
        
        #  place the plane takes off
        self.vertiport_station = {0:[0,0],1:[10,10],2:[54, 54]}
        self.vertiport_station_num = 3
        #  place the plane landed
        # map size
        self.map_len =  map_size[0]
        self.map_information = map_imformaton
        # 比例尺 1h 0.5公里 行程在 1h内， 60 * 60 

        # 每条边 增加拥堵系数，改为路网图# 宏观仿真
