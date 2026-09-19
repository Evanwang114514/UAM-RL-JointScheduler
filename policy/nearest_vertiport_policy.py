import random
from policy.base_policy import BasePolicy


class NearestVertiportPolicy(BasePolicy):
    """
    乘客选择地面时间最近的出发 vertiport
    """

    def __init__(
        self,
        candidate_from_vertiports,
        to_vertiport,
        greedy_prob=1.0
    ):
        """
        Args:
            candidate_from_vertiports: list[int]
                可选的出发 vertiport
            to_vertiport: int
                目标 vertiport
            greedy_prob: float
                选择最近 vertiport 的概率（<1 时引入随机性）
        """
        self.candidate_from_vertiports = candidate_from_vertiports
        self.to_vertiport = to_vertiport
        self.greedy_prob = greedy_prob

    def _select_nearest_vertiport(self, env, pid):
        """
        选择地面行驶时间最近的 vertiport
        """
        person = env.persons.persons[pid]

        best_vid = None
        best_time = float("inf")

        for vid in self.candidate_from_vertiports:
            vid = str(vid)
            vertiport = env.vertiports.vertiport_list[vid]

            t_drive = env.vehicles.estimate_travel_time(
                origin=person.origin_position,
                destination=vertiport.vertiport_position
            )

            if t_drive < best_time:
                best_time = t_drive
                best_vid = vid

        return best_vid

    def act(self, env, state):
        """
        输出 actions
        """
        actions = {}

        waiting_pids = state["waiting_decisions"]

        for pid in waiting_pids:
            # ε-greedy（可选）
            if random.random() < self.greedy_prob:
                from_vid = self._select_nearest_vertiport(env, pid)
            else:
                from_vid = random.choice(self.candidate_from_vertiports)

            actions[pid] = {
                "mode": "UAM",
                "from_vertiport": from_vid,
                "to_vertiport": self.to_vertiport
            }

        return actions
