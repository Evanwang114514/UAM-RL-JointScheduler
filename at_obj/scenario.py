"""
Author: PangAY
Date: 2023-12-08 17:01:38
LastEditTime: 2025-12-16
LastEditors: pangay 1623253042@qq.com
"""

# 统计 passenger 的不同部分的时间 即等待多久，地面多久，等待起飞多久


import random
from at_obj.evtol.vehicle_state import VehicleState

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[SCENARIO_PASSENGER] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scenario_passenger.log", mode="w")
    ]
)


# 因为是立体的后续要加上 eVTOL 的飞行高度，和不同方向的飞行速度
logger = logging.getLogger("UAM_SCENARIO")


import gymnasium as gym
from typing import Dict, List, Any, Optional


from at_obj.person.person_builder import PersonBuilder
from at_obj.vehicle.vehicle_builder import VehicleBuilder
from at_obj.vertiport.vertiport_builder import VertiportBuilder
from at_obj.evtol.evtol_builder import eVTOLBuilder
from at_obj.evtol.evtol import eVTOL
from at_obj.evtol.evtol_registry import EVTOL_SPECS


# 这里 scenario 中增加， 如果飞走了一辆 eVTOL  那么再随机生成一辆降落 保证机场的容量 随机这辆 eVTOL 的需要充电的时间 
# 这里 eVTOL 是固定型号 载客 2 人  飞行速度 200 km/h  续航 50 km 充电需要 25 min # 可调节 # 是文档读取的定义 而不是直接更改文件
# 更改 初始化的时候再对 vertiport 进行 eVTOL 补充 而不要在 VertiportBuilder 中进行补充
# 乘客读取通过 文件读取, 当一定时间时， 乘客不再生成，仿真最后结束
class Scenario(gym.Env):
    """
    Pure simulation environment for UAM + ground transport.
    - Each step = 1 min simulation
    - RL decisions are applied only to new passengers
    - Supports random or file-based passenger generation
    """

    def __init__(
        self,
        max_time: int = 10000,
        passenger_generation_end_time: int = 300,
        person_spawn_file: Optional[str] = None,
        num_evtols_per_vertiport: int = 2,
        enable_logger: bool = True,
    ):
        super().__init__()

        self.max_time = max_time
        self.passenger_generation_end_time = passenger_generation_end_time # 乘客生成截止时间
        self.time = 0

        # Builders
        self.persons = PersonBuilder(spawn_file = person_spawn_file)
        self.vehicles = VehicleBuilder()
        self.vertiports = VertiportBuilder(num_evtols_per_vertiport = num_evtols_per_vertiport)
        self.eVTOL_last_id = 0
        total_evtols = sum(
                        len(v) for v in self.vertiports.evtols_at_vertiport.values()
                            )
        self.eVTOL_last_id  += total_evtols
        self._all_evtols: Dict[str, eVTOL] = {}  # Scenario 全局 EVTOL 管理

        self.vertiport_charge_power = {
                "0": 20,   #   快充
                "1": 30,   #   中等
                #"2": 350,   # kW  中等 
            }


        self.evtols = eVTOLBuilder(
            vertiport_ids=[str(i) for i in range(self.vertiports.vertiport_num)]
        )

                # 记录每个乘客的行程记录
        # 格式: { person_id: [ {"mode":"GROUND|UAM", "start_time":t0, "end_time":t1, "from":vid1, "to":vid2}, ... ] }
        self.person_travel_records: Dict[str, List[Dict[str, Any]]] = {}


        # List of passenger IDs waiting for RL decision
        self.waiting_decisions: List[str] = []

        # Finished passengers
        self.finished_ids: List[str] = []

        self.enable_logger = enable_logger
        if not self.enable_logger:
            logger.setLevel(logging.CRITICAL)  # 只打印严重错误，屏蔽 info/debug

    # =========================
    # Gym API
    # =========================
    def reset(self, seed=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(int(seed))

        self.time = 0
        self.persons.reset()

        # IDs repeat for the fixed trace; never carry travel records across episodes.
        self.person_travel_records = {}
        for pid in self.persons.persons.keys():
            if pid not in self.person_travel_records:
                self.person_travel_records[pid] = []

        self.vehicles.reset()
        self.vertiports.reset()
        self.evtols.reset()

        self._all_evtols.clear()
        self.eVTOL_last_id = 0

        self.waiting_decisions.clear()
        self.finished_ids.clear()


        # ⭐ 关键：t=0 时先补满机场容量
        self._maintain_evtol_capacity()

        logger.info("event = reset init_evtols_completed")

        return self.get_state()
    

    def step(self, action=None):
        """
        One step = 1 minute simulation
        Passenger states are ONLY updated here.
        """
        
        logger.info(f"===== STEP {self.time} START =====")

        # Person-minutes in [t, t+1): every already-generated unfinished passenger counts.
        active_count_start = sum(
            1
            for person in self.persons.persons.values()
            if str(getattr(person, "state", "")).lower() != "finished"
        )
        reward = -float(active_count_start)

        # 1. Spawn 新乘客

        new_persons = []   #  防御性初始化

        if self.time <= self.passenger_generation_end_time:
            new_persons = self.persons.spawn(self.time)
            self.waiting_decisions.extend(new_persons)

            for pid in new_persons:
                logger.info(f"[SPAWN] pid={pid} time={self.time}")


        for pid in new_persons:
            logger.info(f"[SPAWN] pid={pid} time={self.time}")

        if action is not None:
            for pid, act in action.items():
                #person = self.persons.persons[pid]

                # 只允许在 waiting_decisions 中决策
                if pid not in self.waiting_decisions:
                    continue
                
                self.apply_decision(pid, act)
                # person.apply_action(act, self.time)

        # =========================
        # 1. 推进所有系统状态（不改 passenger）
        # =========================
        self.vehicles.update_objects_state(self.time)
        self.vertiports.update_objects_state(self.time) #现在是 vertiport 中进行维护
        self.evtols.update_objects_state(self.time)

        #处理 eVTOL 起飞

        self._handle_evtol_departures()


        # =========================
        # 2. 处理 eVTOL 到达事件（核心）
        # =========================

        for pid, person in self.persons.persons.items():
            
            self.persons.persons[pid].update(self.time)
        
            if  self.persons.persons[pid].sub_state == "arriver_vertiport":

                person = self.persons.persons[pid]

                vid = self.persons.persons[pid].origin_vertiport_id
                self.vertiports.add_new_passenger(
                    person_id = pid,
                    vertiport_id = vid,
                )

                logger.info(
                    "[PASSENGER_ARRIVE_VERTIPORT] "
                    f"pid={pid} vertiport={vid} time={self.time}"
                )


        arrived_events = self._handle_evtol_arrivals()
        

        for evtol_id, passenger_ids, vertiport_id in arrived_events:
            for pid in passenger_ids:
                person = self.persons.persons[pid]

                # passenger 状态更新
                person.state = "finished"
                person.end_time = self.time

                # travel record 更新
                records = self.person_travel_records[pid]
                records[-1]["end_time"] = self.time

                self.finished_ids.append(pid)

                logger.info(
                    f"[ARRIVAL] pid={pid} "
                    f"by_evtol={evtol_id} "
                    f"at_vertiport={vertiport_id} "
                    f"time={self.time}"
                )

        # =========================
        # 3. eVTOL capacity & charging
        # =========================
        self._maintain_evtol_capacity()



        # =========================
        # 5. 日志（此时状态已稳定）
        # =========================
        for pid, person in self.persons.persons.items():
            records = self.person_travel_records.get(pid, [])
            last = records[-1] if records else {}

            logger.info(
                f"[Passenger] id={pid} time={self.time} "
                f"state={person.state} "
                f"sub_state={person.sub_state} " # sub state 没有得到更新
                f"from={last.get('from')} "
                f"to={last.get('to')} "
                f"start={last.get('start_time')} "
                f"end={last.get('end_time')}"
            )

            stats = person.time_stats

            logger.info(
                "[PASSENGER_STATS] "
                f"pid={pid} "
                f"to_vertiport_time={stats['to_vertiport']} "
                f"wait_uam_time={stats['wait_uam']} "
                f"fly_time={stats['fly']} "
                f"total={sum(stats.values())}"
            )


        # =========================
        # 6. Episode termination
        # =========================
        # Spawn logic is inclusive (time <= passenger_generation_end_time).
        generation_done = self.time >= self.passenger_generation_end_time

        n_spawned = len(self.persons.persons)
        n_finished = len(set(self.finished_ids))

        all_finished = (
            generation_done
            and n_spawned > 0
            and n_finished >= n_spawned
            and len(self.waiting_decisions) == 0
        )

        # Natural task success: every generated passenger has completed.
        terminated = bool(all_finished)

        # max_time is ONLY a safety/deadlock guard.
        truncated = bool(self.time >= self.max_time and not terminated)

        info = {
            "generation_done": bool(generation_done),
            "n_spawned": int(n_spawned),
            "n_finished": int(n_finished),
            "completion_rate": (
                float(n_finished) / float(n_spawned)
                if n_spawned > 0 else 0.0
            ),
            "hard_guard_hit": bool(truncated),
        }

        logger.info(
            f"[STEP END] time={self.time} reward={reward} "
            f"finished={n_finished}/{n_spawned} "
            f"terminated={terminated} truncated={truncated}"
        )

        self.time += 1

        return self.get_state(), reward, terminated, truncated, info


    # =========================
    # RL Decision Interface
    # =========================

    def apply_decision(self, person_id: str, decision: Dict[str, Any]):
        """
        decision = {
            "mode": "UAM" | "GROUND",
            "from_vertiport": int,
            "to_vertiport": int
        }
        """
        person = self.persons.persons[person_id]

        if decision["mode"] == "UAM":
            # 分配给 eVTOL
            from_v = str(decision["from_vertiport"])
            to_v = str(decision["to_vertiport"])

            person = self.persons.persons[person_id]

            person.state = "enroute" 
            person.method = "UAM"
            # ===== 1. 记录 Passenger 决策 =====
            person.choose_vertiport(
                from_vertiport_id=from_v,
                to_vertiport_id=to_v,
                current_time = self.time
            )

            # ===== 2. Scenario 计算物理时间 =====
            t_pickup = self.vehicles.estimate_travel_time(
                origin=person.origin_position,
                destination=self.vertiports.vertiport_list[from_v].vertiport_position
            ) # 在本任务中，我们并不考虑路况影响，使用 estimate_travel_time 作为 to vertiport 的时间
       

            person.t_drive_pickup = t_pickup
            person.current_timer = t_pickup # 这里是有问题的，更新混乱

            # ===== Scenario 级行程记录（用于统计 / reward）=====
            self.person_travel_records.setdefault(person_id, []).append({
                "mode": "UAM",
                "from": from_v,
                "to": to_v,
                "start_time": float(
                    getattr(person, "spawn_time", self.time)
                ),
                "end_time": None
            })

            # ===== 统一、正则化日志（强烈推荐）=====
            logger.info(
                "[PASSENGER_DECISION] "
                f"pid={person_id} "
                f"mode=UAM "
                f"from={from_v} "
                f"to={to_v} "
                f"t={self.time} "
                f"pickup_time={t_pickup:.1f}"
            )

        # ===== 从等待决策队列中移除 =====
        if person_id in self.waiting_decisions:
            self.waiting_decisions.remove(person_id)


    # =========================
    # Observation Interfaces
    # =========================
    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "persons": self.persons.get_state(),
            "vehicles": self.vehicles.get_state(),
            "vertiports": self.vertiports.get_state(),
            "evtols": self.evtols.get_state(),
            "waiting_decisions": self.waiting_decisions.copy(),
        }

    def get_person_obs(self, person_id: str) -> Dict[str, Any]:
        return {
            "time": self.time,
            "person": self.persons.persons[person_id].get_observation(),
            "vertiports": self.vertiports.get_state(),
            "evtols": self.evtols.get_state(),
        }
    

    def _maintain_evtol_capacity(self):
        for vertiport_id, vertiport in self.vertiports.vertiport_list.items():
            # 当前空闲或充电状态的 EVTOL
            current_evtols = [
                ev for ev in self._all_evtols.values()
                if ev.current_vertiport_id == vertiport_id
                and ev.state.name in ["IDLE", "CHARGING"]
            ]
            capacity = self.vertiports.vertiport_evtol_capacity.get(vertiport_id, 1)
            num_to_spawn = capacity - len(current_evtols)
                # 🔹 日志：容量检查
            logger.info(
                "[SCENARIO_EVTOL] event=capacity_check "
                f"time={self.time} vertiport={vertiport_id} "
                f"capacity={capacity} current={len(current_evtols)} need_spawn={num_to_spawn}"
            )

            for _ in range(num_to_spawn):
                evtol_id = f"new_{self.eVTOL_last_id}"
                self.eVTOL_last_id  = self.eVTOL_last_id + 1 
                self._spawn_landing_evtol(evtol_id, vertiport_id)

    
    def _handle_evtol_departures(self):
        for evtol in self._all_evtols.values():

            if (evtol.just_departed == False) and (evtol.state == VehicleState.FLYING):
                
                # evtol 的状态好像没有改变
                # doubel check

                evtol.just_departed = True

                for pid in evtol.passenger_ids:
                    person = self.persons.persons[pid]
                    # 🔑 状态更新
                    person.sub_state = "fly"

                    logger.info(
                        "[PASSENGER_BOARD_EVTOL] "
                        f"pid={pid} evtol={evtol.id} time={self.time}"
                    )



    
    def _handle_evtol_arrivals(self):
        """
        Collect eVTOL arrival events.
        DO NOT update passenger states here.
        Returns:
            arrived_events: List of (evtol_id, passenger_ids, vertiport_id)
        """
        arrived_events = []

        for evtol in self._all_evtols.values():

            # 只处理刚刚到达的 eVTOL
            if not getattr(evtol, "just_arrived", False):
                continue

            evtol_id = evtol.id
            dest_vertiport = evtol.current_vertiport_id
            passenger_ids = list(evtol.passenger_ids)  # ⚠️ 必须从 evtol 本体取
            assert evtol.passenger_ids, (
                         f"EVTOL {evtol.id} arrived with EMPTY passenger list!" )

            for pid in evtol.passenger_ids:
                person = self.persons.persons[pid]
                    # 🔑 状态更新
                person.state = "finished"
                person.sub_state = "arrived"

            arrived_events.append(
                (evtol_id, passenger_ids, dest_vertiport)
            )

            logger.info(
                "[EVTOL_ARRIVAL_EVENT] "
                f"evtol={evtol_id} "
                f"passengers={passenger_ids} "
                f"to_vertiport={dest_vertiport} "
                f"time={self.time}"
            )

            # ===== 清理 eVTOL 状态（但不管 passenger）=====
            evtol.passenger_ids.clear()
            evtol.just_arrived = False
            evtol.state = VehicleState.IDLE   # 或 CHARGING，由后续逻辑决定

        return arrived_events



    def _spawn_landing_evtol(self, evtol_id: str , vertiport_id: str):
        """
        Spawn a new EVTOL with a specific ID and register it globally.
        """
        # 使用 eVTOLBuilder 的接口，传入指定 ID
       
        evtol = self.evtols.spawn_replacement_evtol(evtol_id=evtol_id, vertiport_id=vertiport_id)
         
    
        # 初始化状态
        evtol.state = evtol.state.CHARGING
        evtol.battery_kwh = evtol.spec.battery_capacity_kwh * random.uniform(0.1, 0.3)  # 初始电量 20%-50%

        # 注册到全局 EVTOL 列表
        self._all_evtols[evtol.id] = evtol

        # 同步到 Vertiport
        self.vertiports.evtols_at_vertiport[vertiport_id].append(evtol)

        logger.info(
            "[SCENARIO_EVTOL] event=spawn "
            f"time={self.time} "
            f"evtol_id={evtol.id} "
            f"vertiport={vertiport_id} "
            f"state={evtol.state.name} "
            f"battery={evtol.battery_kwh:.2f} "
            f"capacity={evtol.spec.capacity}"
        )
    
    # 此函数要通过 vehicle 实现，现在地面车辆并没有很好的进入仿真
    def get_drive_time(self, begin: List[int], end: List[int], rate_ratio: float = 1.0) -> float:
        """
        Calculate Manhattan travel time in minutes.
        """
        distance = abs(begin[0] - end[0]) + abs(begin[1] - end[1])
        travel_time = distance / (self.speed * rate_ratio)
        return travel_time