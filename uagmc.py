'''
Author: pangay 1623253042@qq.com
Date: 2024-01-11 20:58:46
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2025-05-13 14:57:04
'''
import math
import random
from typing import Dict, List, Optional

from at_obj.map.map import Map
from at_obj.vertiport.vertiport import VertiportInfo
from at_obj.evtol.evtol import EVTOL  # 假设你有这个类


class VertiportBuilder:
    """
    Manage UAM vertiports and EVTOL resources.
    Handles:
    - Passenger queues
    - EVTOL fleet at each vertiport
    - Charging and availability
    - Flight scheduling
    """

    def __init__(self) -> None:
        self.map = Map()
        self.vertiport_num = self.map.vertiport_station_num
        self.vertiport_list: Dict[int, VertiportInfo] = {}
        self.time = 0
        self.init_builder()

    def init_builder(self):
        """Initialize or reset vertiports."""
        self.vertiport_list.clear()
        self.time = 0
        for vid in range(self.vertiport_num):
            position = self.map.vertiport_station[vid]
            vertiport = VertiportInfo(id=vid, vertiport_position=position)
            # EVTOL fleet
            vertiport.evtol_list: List[EVTOL] = []  # 所有 EVTOL
            vertiport.evtol_capacity = 3  # 默认每个 vertiport 可同时停放 EVTOL 数量
            vertiport.charging_speed = 1.0  # 充电速度单位：百分比 / 分钟
            self.vertiport_list[vid] = vertiport

    def reset(self):
        """Reset builder to initial state (for Scenario.reset)."""
        self.init_builder()

    # ========================
    # EVTOL 相关
    # ========================
    def add_evtol(self, evtol: EVTOL, vertiport_id: int):
        """Add an EVTOL to the vertiport fleet."""
        self.vertiport_list[vertiport_id].evtol_list.append(evtol)

    def get_available_evtol(self, vertiport_id: int) -> Optional[EVTOL]:
        """Return the next available EVTOL at vertiport, or None if none available."""
        for e in self.vertiport_list[vertiport_id].evtol_list:
            if e.is_available():  # EVTOL 类需要有 is_available() 方法
                return e
        return None

    def update_evtol_state(self, time: int):
        """Update EVTOLs: charging, flight countdown, availability."""
        self.time = time
        for vertiport in self.vertiport_list.values():
            for evtol in vertiport.evtol_list:
                evtol.update_state(time)  # EVTOL 类需提供 update_state 方法，处理飞行/充电

    def get_next_evtol_info(self, vertiport_id: int) -> Optional[Dict]:
        """Get info of the next EVTOL that can take off."""
        evtol = self.get_available_evtol(vertiport_id)
        if evtol is None:
            return None
        return {
            "id": evtol.id,
            "charge": evtol.charge_level,
            "state": evtol.state,
            "speed": evtol.speed
        }

    # ========================
    # Passenger 队列管理
    # ========================
    def add_new_passenger(self, person_id: str, vertiport_id: int):
        if vertiport_id in self.vertiport_list:
            vertiport = self.vertiport_list[vertiport_id]
            vertiport.person_list.append(person_id)
            vertiport.wait_person += 1

    def cal_distance(self, p1: List[int], p2: List[int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def get_fly_time(self, origin: List[int], destination: List[int], vertiport_id: int) -> int:
        """Return flight time from origin to destination."""
        distance = self.cal_distance(origin, destination)
        vertiport = self.vertiport_list[vertiport_id]
        return max(1, int(distance / vertiport.speed))

    def update_objects_state(self, time: int):
        """Update vertiports' passenger queues and volume for the current time step."""
        self.time = time
        volume_list = [3, 2]  # example capacity for first two vertiports
        for vid, vertiport in self.vertiport_list.items():
            # update passenger queue
            if vid < len(volume_list):
                vertiport.volume = max(0, volume_list[vid] + random.randint(-1, 0))
                served = min(vertiport.wait_person, vertiport.volume)
                vertiport.wait_person -= served
                vertiport.now_volume = vertiport.volume
        # update EVTOL state
        self.update_evtol_state(time)

    def get_all_wait_person(self) -> List[int]:
        """Return list of waiting passengers for all vertiports."""
        return [v.wait_person for v in self.vertiport_list.values()]

    def get_wait_person(self, vertiport_id: int) -> int:
        return self.vertiport_list[vertiport_id].wait_person

    def get_state(self) -> Dict[int, VertiportInfo]:
        return self.vertiport_list
