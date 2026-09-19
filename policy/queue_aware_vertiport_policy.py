"""
Author: PangAY
Queue-aware vertiport selection policy
"""

from typing import Dict, Any, List
import random
import math


class QueueAwareVertiportPolicy:
    def __init__(
        self,
        candidate_from_vertiports: List[int],
        to_vertiport: int,
        greedy_prob: float = 1.0,
        wait_coeff: float = 1.0,
        congestion_coeff: float = 1.0,
    ):
        """
        Args:
            candidate_from_vertiports: 可选起飞 vertiport
            to_vertiport: 固定目的 vertiport
            greedy_prob: 选择最优的概率
            wait_coeff: 等待时间权重
            congestion_coeff: 拥堵惩罚权重
        """
        self.from_candidates = candidate_from_vertiports
        self.to_vertiport = to_vertiport
        self.greedy_prob = greedy_prob
        self.wait_coeff = wait_coeff
        self.congestion_coeff = congestion_coeff

    # =========================
    # Main interface
    # =========================
    def act(self, env, state) -> Dict[str, Dict[str, Any]]:
        actions = {}

        for pid in state["waiting_decisions"]:
            from_vid = self._select_vertiport(env, pid)

            actions[pid] = {
                "mode": "UAM",
                "from_vertiport": from_vid,
                "to_vertiport": self.to_vertiport,
            }

        return actions

    # =========================
    # Core logic
    # =========================
    def _select_vertiport(self, env, person_id: str) -> int:
        """
        Choose vertiport with minimal estimated total cost
        """

        # exploration
        if random.random() > self.greedy_prob:
            return random.choice(self.from_candidates)

        person = env.persons.persons[person_id]
        origin = person.origin_position

        best_vid = None
        best_cost = float("inf")

        for vid in self.from_candidates:
            vertiport = env.vertiports.vertiport_list[str(vid)]

            # ---------- 1. 地面到达时间 ----------
            drive_time = env.vehicles.estimate_travel_time(
                origin=origin,
                destination=vertiport.vertiport_position,
            )

            # ---------- 2. 当前排队等待 ----------
            queue_len = len(vertiport.person_list)


            # 可用 eVTOL 数
            available_evtols = [
                ev for ev in vertiport.evtol_list
                if ev.state.name in ["IDLE", "CHARGING"]
            ]
            service_capacity = max(len(available_evtols), 1)

            expected_wait = queue_len / service_capacity

            # ---------- 3. 拥堵惩罚（软约束） ----------
            congestion_penalty = math.log(queue_len + 1)

            # ---------- 4. 总 cost ----------
            total_cost = (
                drive_time
                + self.wait_coeff * expected_wait
                + self.congestion_coeff * congestion_penalty
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_vid = vid

        return best_vid
