'''
Author: PangAY
Date: 2023-12-08 20:51:05
LastEditTime: 2024-01-20 22:55:27
LastEditors: pangay 1623253042@qq.com
'''
from typing import List, Dict, Any


class PersonInfo:
    """
    Passenger entity for UAM scenario.
    Passive state machine.
    """

    def __init__(
        self,
        id: str,
        spawn_time: int,
        origin_position: List[int],
        destination_position: List[int]
    ):
        # ========= Identity =========
        self.id = id

        # ========= Spatial =========
        self.origin_position = origin_position
        self.destination_position = destination_position

        # ========= Time =========
        self.spawn_time = spawn_time
        self.decision_time = None
        self.assign_time = None
        self.arrival_time = None
        self.finish_time = None
        self.origin_vertiport_id = None
        self.destination_vertiport_id = None

        # ========= Decision =========
        self.method = None           # "ground" | "UAM"
        self.action: Dict[str, Any] = {}

        # ========= Travel segments =========
        self.t_drive_pickup = 0 # 在分配 vertiport 之后进行更新
        self.t_wait_uam = 0
        self.t_fly = 0
        self.t_drive_dest = 0

        # ========= Runtime =========
        self.state = "new"            # new / assigned / enroute / finished / removed
        self.sub_state = None         # v / w / a / vd ...
        self.current_timer = 0

        self.time_stats = {
            "to_vertiport": 0,
            "wait_uam": 0,
            "fly": 0,
        }

        self.vehicle_id = None

        # ========= Logs =========
        self.state_trace = []

    # =========================
    # Decision interface
    # =========================
    def assign(self, method: str, action: Dict[str, Any], time: int):
        """
        Called by Scenario / Policy
        """
        self.method = method
        self.action = action
        self.decision_time = time
        self.assign_time = time
        self.state = "assigned"

    # =========================
    # State update
    # =========================
    # State 一共有几个状态  new 没有被分配  ground to vertiport 
    def update(self, time: int):
        self.state_trace.append((time, self.state, self.sub_state))

        if self.sub_state in self.time_stats:
            self.time_stats[self.sub_state] += 1

        if self.state == "assigned": # 任务或航班已经被分配给 eVTOL
            self._enter_enroute()

        elif self.state == "enroute": # 乘客正在前往目的地的途中
            self._update_enroute()

        elif self.state == "finished": # 乘客已经到达目的地
            self.finish_time = time
            self.state = "removed"

    # =========================
    # Internal FSM
    # =========================
    def _enter_enroute(self):

        self.state = "enroute"

        if self.method == "ground":
            self.sub_state = "drive"
            self.current_timer = self.t_drive_dest

        elif self.method == "UAM":
            self.sub_state = "to_vertiport"
            self.current_timer = self.t_drive_pickup
        
    #只做状态更新
    def _update_enroute(self):

        if self.current_timer is None:
            self.current_timer = self.t_drive_pickup
            return None # 防御性检查

        if self.current_timer > 0:
            self.current_timer -= 1
            return

        if self.method == "UAM":
            if self.sub_state == "to_vertiport":
                # ✅ 到达起飞 vertiport
                self.sub_state = "arriver_vertiport"
                self.current_timer = self.t_wait_uam

                # 🔑 此刻才真正加入 vertiport 队列
                # 此功能 在 scenario 中进行处理
                # self.vertiports.vertiport_list[
                #     self.pending_vertiport_id 
                # ].add_new_passenger(self.id)

            elif self.sub_state == "arriver_vertiport":
                self.sub_state = "wait_uam"

                pass


    def _finish(self):
        self.state = "finished"
        self.arrival_time = self.spawn_time + self.total_travel_time()

    # =========================
    # Utility
    # =========================
    def total_travel_time(self):
        return (
            self.t_drive_pickup +
            self.t_wait_uam +
            self.t_fly +
            self.t_drive_dest
        )
        
    def plan_uam_trip(
        self,
        from_vertiport_id: str,
        to_vertiport_id: str,
        vertiports,
    ):
        """
        Decide all travel segment durations for UAM trip
        """

        # ===== Ground to pickup vertiport =====
        # 这个是 vehicle driving time from origin to vertiport，应该使用 vehicle
        self.t_drive_pickup = self._calc_distance(
            self.origin,
            from_vertiport_id
        )

        # ===== Waiting time at vertiport (initial expectation) =====
        # ⚠️ 这是期望值，不是最终值
        self.t_wait_uam = vertiports.get_expected_wait_time(
            from_vertiport_id
        ) #因为还有充电等，这个地方 timer 直接用真实数值


        # ===== Flight time =====
        dist_km = vertiports.get_air_distance(
            from_vertiport_id,
            to_vertiport_id
        )
        self.t_fly = dist_km / self.uam_speed * 60.0

        # ===== Ground after landing =====
        self.t_drive_dest = self._calc_distance(
            to_vertiport_id,
            self.destination
        )

        # ===== 初始化 FSM timer =====
        self.current_timer = self.t_drive_pickup

    def choose_vertiport(
        self,
        from_vertiport_id: str,
        to_vertiport_id: str,
        current_time: int
    ):
        """
        ONLY record decision & initialize FSM.
        NO time / distance calculation here.
        """

        self.method = "UAM"

        self.action = {
            "type": "UAM_ROUTE",
            "from_vertiport": from_vertiport_id,
            "to_vertiport": to_vertiport_id
        }

        self.origin_vertiport_id = from_vertiport_id
        self.destination_vertiport_id = to_vertiport_id

        self.decision_time = current_time
        self.assign_time = current_time

        # FSM
        self.state = "enroute"
        self.sub_state = "to_vertiport"

        # timers 由 Scenario 填
        self.current_timer = None
        self.pending_vertiport_id = from_vertiport_id




    def _calc_distance(self, origin, destination):
        # 简单欧几里得距离
        dx = destination[0] - origin[0]
        dy = destination[1] - origin[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def get_observation(self):
        """
        Used by RL wrapper
        """
        return {
            "origin": self.origin_position,
            "destination": self.destination_position,
            "state": self.state,
            "sub_state": self.sub_state,
            "method": self.method,
            "action": self.action
        }
