"""
VertiportBuilder.py
Author: PangAY
Date: 2023-12-08
LastEditTime: 2025-12-16
Description: UAM Vertiport manager for passengers and eVTOLs
"""

import math
import random
from typing import Dict, List, Any, Optional

from at_obj.map.map import Map
from at_obj.vertiport.vertiport import VertiportInfo
from at_obj.evtol.evtol_builder import eVTOLBuilder 
from at_obj.evtol.evtol import eVTOL


class VertiportBuilder:
    """
    Manages all vertiports in the map.
    Responsibilities:
    - Initialize vertiports
    - Track passengers waiting at each vertiport
    - Manage eVTOLs at each vertiport
    - Update vertiport state per timestep
    """

    def __init__(self, num_evtols_per_vertiport: int = 2) -> None:
        self.map = Map()
        self.vertiport_list: Dict[str, VertiportInfo] = {}
        self.vertiport_num = self.map.vertiport_station_num
        self.time = 0

        # 初始化 eVTOLs 存储
        self.num_evtols_per_vertiport = num_evtols_per_vertiport
                # 默认每个 vertiport 2 架
        
        self.vertiport_evtol_capacity = {
                                    "0": 3, # 测试场景 vertiport 0 有 3 架 即城市郊区有更大的机场
                                    "1": 1,
                                    "2": 1
                                }

        # if vertiport_evtol_capacity is None:
        #     vertiport_evtol_capacity = {
        #         str(i): 2 for i in range(self.vertiport_num)
        #     }
        
        # 初始化的eVTOL 通过 scenario
        # self.evtol_builder = eVTOLBuilder(
        # vertiport_ids=list(self.vertiport_evtol_capacity.keys())
        #     )

        self.evtols_at_vertiport: Dict[str, List[eVTOL]] = {} #要在 vertiport 中维护

        # 初始化 vertiport 对象
        self._init_vertiports()

    # ========================
    # 初始化方法
    # ========================
    # vertiport 不在这里进行增补 
    def _init_vertiports(self):
        self.vertiport_list.clear()
        self.evtols_at_vertiport.clear()

        for vid in range(self.vertiport_num):
            vid_str = str(vid)
            vertiport_position = self.map.vertiport_station[vid]

            self.vertiport_list[vid_str] = VertiportInfo(
                id=vid,
                vertiport_position=vertiport_position,
            )

            self.evtols_at_vertiport[vid_str] = []

    # def _init_evtol_fleet(self):
    #     """统一创建 eVTOL fleet，并按容量分配到 vertiports"""
    #     # 清空各 vertiport 的 eVTOL 列表
    #     for evtol_list in self.evtols_at_vertiport.values():
    #         evtol_list.clear()

    #     for vid, num_evtol in self.vertiport_evtol_capacity.items():
    #         if num_evtol <= 0:
    #             continue

    #         for i in range(num_evtol):
    #             # 生成唯一 EVTOL ID，例如 ev0_0, ev0_1, ev1_0...
    #             evtol_id = f"ev{vid}_{i}"

    #             evtol = self.evtol_builder.spawn_evtol(
    #                 evtol_id=evtol_id,
    #                 vertiport_id=vid
    #             )

    #             self.evtols_at_vertiport[vid].append(evtol)


    def create_objects(self, vertiport_id: int):
        """创建单个 vertiport 对象"""
        vertiport_position = self.map.vertiport_station[vertiport_id]
        vertiport_info = VertiportInfo(
            id=vertiport_id,
            vertiport_position=vertiport_position,
        )
        self.vertiport_list[str(vertiport_id)] = vertiport_info

    # ========================
    # 乘客管理
    # ========================
    def add_new_passenger(self, person_id: str, vertiport_id: int):
        """添加乘客到指定 vertiport"""
        v = self.vertiport_list[str(vertiport_id)]
        v.person_list.append(person_id)
        v.wait_person += 1
        v.total_person += 1

    def get_waiting_passengers(self, vertiport_id: int) -> List[str]:
        return self.vertiport_list[str(vertiport_id)].person_list

    def get_all_wait_person(self) -> List[int]:
        return [v.wait_person for v in self.vertiport_list.values()]

    # ========================
    # eVTOL 管理
    # ========================
    def get_available_evtols(self, vertiport_id: int) -> List[eVTOL]:
        """返回可以起飞的 eVTOL"""
        return [e for e in self.evtols_at_vertiport[str(vertiport_id)] if e.is_available()]

    def assign_evtol_flight(
        self,
        vertiport_id: int,
        evtol: eVTOL,
        dest_vertiport_id: int,
        distance_km: float,
        flight_time_min: float,
        passenger_count: int
    ):
        """指派 eVTOL 飞行"""
        evtol.assign_flight(dest_vertiport_id, distance_km, flight_time_min, passenger_count)

    def update_objects_state(self, time: int):
        """
        更新 vertiport 和 eVTOL 状态
        服务乘客仅通过 eVTOL 起飞完成（离散事件）
        """
        self.time = time

        # =========================
        # 1. Vertiport -> eVTOL 调度
        # =========================
        for vertiport_id, vertiport in self.vertiport_list.items():

            available_evtols = self.get_available_evtols(vertiport_id)

            for e in available_evtols:

                # ===== 1. 是否坐满 =====
                if vertiport.wait_person < e.spec.capacity:
                    continue

                # ===== 2. 选择目的地（此处可由 RL 决定）=====
                dest_vertiport_id = "2" # 这次的场景固定 vertiport 的目的地，即固定航线

                distance_km = self._calculate_distance(
                    vertiport.vertiport_position,
                    self.vertiport_list[dest_vertiport_id].vertiport_position
                )

                required_energy = distance_km * e.spec.energy_consumption_kwh_per_km

                # ===== 3. 电量检查 =====
                if e.battery_kwh < required_energy:
                    continue

                # ===== 4. 绑定乘客 =====
                passenger_ids = vertiport.person_list[:e.spec.capacity]
                vertiport.person_list = vertiport.person_list[e.spec.capacity:]

                vertiport.wait_person -= e.spec.capacity
                vertiport.leave_person += e.spec.capacity

                # ===== 5. 更新 eVTOL 状态 =====
                e.start_flight(
                    dest_vertiport_id = dest_vertiport_id,
                    distance_km = distance_km,
                    passenger_ids = passenger_ids
                )

                # 起飞后移出 vertiport
                self.evtols_at_vertiport[vertiport_id].remove(e)

                break  # 一个 timestep 一架


        # =========================
        # 2. 推进所有 eVTOL 状态
        # =========================
        for evtol_list in self.evtols_at_vertiport.values():
            for e in evtol_list:
                e.step(1.0)  # 1 minute per step

        # 更新电量
        self.charge_evtols_at_vertiport(time_step_min=1.0)

    
    #公有函数可以放在一个 函数库中
    def _calculate_distance(self, p1: List[int], p2: List[int]) -> float:
        """计算两点欧氏距离"""
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def charge_evtols_at_vertiport(self, time_step_min: float = 1.0):
        """
        给每个 vertiport 内的 EVTOL 充电
        Args:
            time_step_min: 当前 timestep 的分钟数
        """
        for vertiport_id, evtol_list in self.evtols_at_vertiport.items():
            for e in evtol_list:
                if e.state.name == "CHARGING":
                    # 增加电量
                    e.battery_kwh += e.spec.charge_rate_kwh_per_min * time_step_min
                    # 不超过最大容量
                    e.battery_kwh = min(e.battery_kwh, e.spec.battery_capacity_kwh)
                    
                    # 电量充满后改为 IDLE
                    if e.battery_kwh >= e.spec.battery_capacity_kwh:
                        e.state = e.state.IDLE

    # ========================
    # 状态接口
    # ========================
    def get_state(self):
        state = {}
        for vid, v in self.vertiport_list.items():
            state[vid] = {
                "position": v.vertiport_position,
                "wait_person": v.wait_person,
                "total_person": v.total_person,
                "now_volume": v.now_volume,
                "person_list": list(v.person_list), # 新增乘客列表
                "evtols": [
                    {
                        "id": e.id,
                        "state": e.state.name,
                        "battery_pct": e.battery_pct,
                        "passengers": e.passenger_ids,
                        "current_vertiport": e.current_vertiport_id
                    }
                    for e in self.evtols_at_vertiport[vid]
                ]
            }
        return state

    # ========================
    # 重置方法
    # ========================
    def reset(self):
        self.time = 0
        self._init_vertiports()  
        # self._init_evtol_fleet()
        for vertiport in self.vertiport_list.values():
            vertiport.wait_person = 0
            vertiport.person_list = []
            vertiport.now_volume = 0
            vertiport.leave_person = 0
            vertiport.total_person = 0

