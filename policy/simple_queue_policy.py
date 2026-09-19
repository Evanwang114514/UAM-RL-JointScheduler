"""
Author: PangAY
Queue-only Vertiport Selection Policy
Only considers the current queue length at each vertiport.
"""

from typing import Dict, Any, List
import random


class QueueOnlyVertiportPolicy:
    def __init__(
        self,
        candidate_from_vertiports: List[int],
        to_vertiport: int,
        greedy_prob: float = 1.0
    ):
        """
        Args:
            candidate_from_vertiports: 可选起飞 vertiport
            to_vertiport: 固定目的 vertiport
            greedy_prob: 以 greedy 方式选择最小队列的概率
        """
        self.from_candidates = candidate_from_vertiports
        self.to_vertiport = to_vertiport
        self.greedy_prob = greedy_prob

    # =========================
    # Main interface
    # =========================
    def act(self, env, state) -> Dict[str, Dict[str, Any]]:
        """
        Generate actions for all waiting passengers
        """
        actions = {}

        for pid in state["waiting_decisions"]:
            from_vid = self._select_vertiport(env)
            actions[pid] = {
                "mode": "UAM",
                "from_vertiport": from_vid,
                "to_vertiport": self.to_vertiport,
            }

        return actions

    # =========================
    # Core logic
    # =========================
    def _select_vertiport(self, env) -> int:
        """
        Select vertiport with minimum current queue length
        """
        # ----- epsilon-style exploration -----
        if random.random() > self.greedy_prob:
            return random.choice(self.from_candidates)

        min_queue = float("inf")
        best_candidates = []

        for vid in self.from_candidates:
            vertiport = env.vertiports.vertiport_list[str(vid)]
            queue_len = len(vertiport.person_list)

            if queue_len < min_queue:
                min_queue = queue_len
                best_candidates = [vid]
            elif queue_len == min_queue:
                best_candidates.append(vid)

        # tie-breaking: random among min-queue vertiports
        return random.choice(best_candidates)
