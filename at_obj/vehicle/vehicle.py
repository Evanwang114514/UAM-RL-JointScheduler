'''
Author: PangAY
Date: 2023-12-08 21:13:26
LastEditTime: 2024-01-22 12:37:57
LastEditors: pangay 1623253042@qq.com
'''
from typing import List, Dict

class Vehicle:
    """
    Vehicle data structure for simulation.
    Supports step-wise state update and passenger assignment.
    """

    def __init__(
        self,
        id: str = '0',
        origin_position: List[int] = [0, 0],
        speed: float = 1.5  # 1.5 grid/min (~45 km/h)
    ) -> None:
        self.id = id
        self.origin_position = origin_position
        self.position = origin_position.copy()
        self.speed = speed

        self.passenger_id: str = None
        self.state: str = "idle"  # idle, drive, done
        self.remaining_time: float = 0  # remaining time to reach destination

    # =========================
    # 任务分配接口
    # =========================
    def assign_passenger(self, passenger_id: str, begin: List[int], end: List[int], rate_ratio: float = 1.0):
        """
        Assign a passenger to this vehicle.
        """
        self.passenger_id = passenger_id
        self.state = "drive"
        self.remaining_time = self.get_drive_time(begin, end, rate_ratio)

    # =========================
    # 按步推进
    # =========================
    def update_state(self):
        """
        Update vehicle state per step (1 min).
        """
        if self.state == "drive":
            self.remaining_time -= 1
            if self.remaining_time <= 0:
                self.state = "done"
                self.passenger_id = None

    # =========================
    # 工具方法
    # =========================
    def get_drive_time(self, begin: List[int], end: List[int], rate_ratio: float = 1.0) -> float:
        """
        Calculate Manhattan travel time in minutes.
        """
        distance = abs(begin[0] - end[0]) + abs(begin[1] - end[1])
        travel_time = distance / (self.speed * rate_ratio)
        return travel_time

    def get_state(self) -> Dict[str, dict]:
        return {
            self.id: {
                "position": self.position.copy(),
                "state": self.state,
                "passenger_id": self.passenger_id,
                "remaining_time": self.remaining_time
            }
        }

    def create_object(self):
        return self
