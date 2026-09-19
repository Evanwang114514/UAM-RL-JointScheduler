"""
Description: Vertiport information with EVTOL management
Author: PangAY
Date: 2023-12-08 21:14:21
LastEditTime: 2024-01-22 13:47:52
LastEditors: pangay 1623253042@qq.com
"""
import math
from typing import List, Dict, Optional

from at_obj.evtol.evtol import eVTOL  

class VertiportInfo:
    """
    Describe a UAM vertiport with EVTOL fleet.
    Includes position, passenger queue, EVTOL management, and flight times.
    """
    def __init__(
        self,
        id: int = 0,
        speed: int = 4,  # 120 km/h
        volume: int = 3,
        vertiport_position: List[int] = [0, 0],
        capacity: int = 3,  # EVTOL capacity at vertiport
        charging_speed: float = 1.0  # charging speed per minute
    ):
        self.id = id
        self.vertiport_position = vertiport_position
        self.vertiport_destination = [54, 54]
        self.speed = speed  # flight speed
        self.volume = volume  # passengers per minute
        self.now_volume = 0

        # Passenger management
        self.wait_person = 0
        self.person_list: List[str] = []
        self.total_person = 0
        self.leave_person = 0
        self.time = 0

        # EVTOL management
        self.evtol_list: List[eVTOL] = []
        self.evtol_capacity = capacity
        self.charging_speed = charging_speed  # % per minute

    # ========================
    # Passenger management
    # ========================
    # add_waiting_person  --> add_new_passenger
    def add_new_passenger(self, person_id: str):
        self.person_list.append(person_id)
        self.wait_person += 1
        self.total_person += 1

    def get_wait_time(self) -> int:
        """Estimate waiting time for current passengers (in minutes)."""
        if self.volume == 0:
            return 0
        return int(self.wait_person / self.volume)

    # ========================
    # EVTOL management
    # ========================
    def add_evtol(self, evtol: eVTOL):
        """Add an EVTOL to the vertiport."""
        if len(self.evtol_list) < self.evtol_capacity:
            self.evtol_list.append(evtol)

    def get_available_evtol(self) -> Optional[eVTOL]:
        """Return the next available EVTOL or None if none available."""
        for e in self.evtol_list:
            if e.is_available():
                return e
        return None

    def get_next_evtol_info(self) -> Optional[Dict]:
        """Return info of the next EVTOL that can take off."""
        evtol = self.get_available_evtol()
        if evtol is None:
            return None
        return {
            "id": evtol.id,
            "charge": evtol.charge_level,
            "state": evtol.state,
            "speed": evtol.speed
        }

    def update_evtol_state(self, time: int):
        """Update all EVTOLs at this vertiport."""
        for evtol in self.evtol_list:
            evtol.update_state(time)

    # ========================
    # Distance & Flight Time
    # ========================
    def cal_distance(self, p1: List[int], p2: List[int]) -> float:
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def get_fly_time(self) -> int:
        """Estimate flight time from this vertiport to its destination."""
        distance = self.cal_distance(self.vertiport_position, self.vertiport_destination)
        return max(1, int(distance / self.speed))

    # ========================
    # State
    # ========================
    def get_state(self):
        wait_time = self.get_wait_time()
        fly_time = self.get_fly_time()
        next_evtol = self.get_next_evtol_info()
        return {
            "vertiport_position": self.vertiport_position,
            "vertiport_destination": self.vertiport_destination,
            "wait_person": self.wait_person,
            "person_list": list(self.person_list),
            "wait_time": wait_time,
            "now_volume": self.now_volume,
            "fly_time": fly_time,
            "next_evtol": next_evtol,
        }
