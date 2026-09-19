'''
Author: PangAY
Date: 2023-12-08 21:33:12
LastEditTime: 2024-01-17 23:37:15
LastEditors: pangay 1623253042@qq.com
'''
import random
import math
from typing import Dict, List, Any
from loguru import logger

from .vehicle import Vehicle
from at_obj.map.map import Map


class VehicleBuilder:
    """
    Vehicle manager.
    Responsible for:
    - spawning new vehicles
    - updating vehicle states
    - providing state info for decision-making
    """

    def __init__(
        self,
        map: Any = Map(),
        init_vehicle_num: int = 10,
        spawn_rate: int = 1,      # 每步新增车辆数量
        spawn_interval: int = 1   # 每隔几分钟生成新车辆
    ) -> None:
        self.map = map
        self.init_vehicle_num = init_vehicle_num
        self.spawn_rate = spawn_rate
        self.spawn_interval = spawn_interval

        self.time = 0
        self.vehicles: Dict[str, Vehicle] = {}

        # 初始化车辆
        for vid in range(self.init_vehicle_num):
            vehicle = self._create_vehicle(str(vid))
            self.vehicles[str(vid)] = vehicle

    # =========================
    # 生命周期
    # =========================
    def reset(self):
        self.time = 0
        self.vehicles.clear()
        for vid in range(self.init_vehicle_num):
            vehicle = self._create_vehicle(str(vid))
            self.vehicles[str(vid)] = vehicle

    # =========================
    # 生成新车辆
    # =========================
    def spawn(self, time: int) -> List[str]:
        """
        每隔 spawn_interval 生成新的车辆
        返回新生成车辆的 ID 列表
        """
        self.time = time
        new_ids = []

        if time % self.spawn_interval != 0:
            return new_ids

        for i in range(self.spawn_rate):
            vid = str(len(self.vehicles) + i)
            vehicle = self._create_vehicle(vid)
            self.vehicles[vid] = vehicle
            new_ids.append(vid)

        return new_ids

    def _create_vehicle(self, vehicle_id: str) -> Vehicle:
        origin_position = [
            random.randint(0, int(0.3 * self.map.map_len)),
            random.randint(0, int(0.3 * self.map.map_len))
        ]
        vehicle = Vehicle(vehicle_id, origin_position=origin_position).create_object()
        vehicle.state = "idle"  # 默认状态为空闲
        return vehicle

    # =========================
    # 状态更新
    # =========================
    def update_objects_state(self, time: int):
        """
        每步更新车辆状态：
        - 移动中的车辆可以被标记删除或完成任务
        - 空闲车辆继续存在
        """
        self.time = time
        finished_ids = []

        for vid, vehicle in list(self.vehicles.items()):
            vehicle.update_state()
            if vehicle.state == "done":  # 完成任务，或者离开环境
                finished_ids.append(vid)
                del self.vehicles[vid]

        return finished_ids

    # =========================
    # 查询接口
    # =========================
    def get_state(self):
        return {vid: v.get_state() for vid, v in self.vehicles.items()}

    def get_idle_vehicles(self) -> List[str]:
        return [vid for vid, v in self.vehicles.items() if v.state == "idle"]


    def estimate_travel_time(
        self,
        origin: List[float],
        destination: List[float],
        speed_kmph: float = 40.0
    ) -> int:
        """
        Estimate ground travel time between two positions.

        Args:
            origin: [x, y]
            destination: [x, y]
            speed_kmph: average ground vehicle speed

        Returns:
            travel_time (int): minutes (>=1)
        """
        dx = (destination[0] - origin[0]) * 0.5  # 地图比例尺 1h 0.5公里
        dy = (destination[1] - origin[1]) * 0.5

        # 欧式距离（km）
        distance_km = math.sqrt(dx * dx + dy * dy)

        # 时间（分钟）
        travel_time_min = distance_km / speed_kmph * 60.0

        # 至少 1 步，避免 0
        return max(1, int(math.ceil(travel_time_min)))