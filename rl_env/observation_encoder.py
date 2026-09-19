import numpy as np


class ObservationEncoder:
    """
    System-level fixed-dimension observation encoder
    (Aggregated eVTOL-aware version)
    """

    def __init__(self, num_vertiports: int):
        self.num_vertiports = num_vertiports

        # passenger: origin(x,y) + destination(x,y)
        self.person_dim = 4

        # per vertiport aggregated features:
        # waiting_cnt + incoming_cnt + charging_evtols + total_evtols +
        # total_capacity + avg_charge_time + min_charge_time + avg_flight_time
        self.vertiport_dim = 8

        self.obs_dim = self.person_dim + num_vertiports * self.vertiport_dim

    def encode(self, env, state):
        """
        Encode current environment state into a fixed-size vector.
        If no waiting passenger, passenger part is zero-padded.
        """

        obs = []

        # ======================
        # 1. Passenger info
        # ======================
        waiting = state["waiting_decisions"]

        if len(waiting) > 0:
            pid = waiting[0]  # decision focus: first waiting passenger
            p = env.persons.persons[pid]
            obs.extend(p.origin_position)
            obs.extend(p.destination_position)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # ======================
        # 2. Vertiport info
        # ======================
        for vid, v in env.vertiports.vertiport_list.items():
            if int(vid) == 2:
                continue

            # =========================
            # Passenger side
            # =========================
            waiting_cnt = len(v.person_list)

            incoming_cnt = 0
            for person in env.persons.persons.values():
                if person.state == "enroute" and str(person.origin_vertiport_id) == vid:
                    incoming_cnt += 1

            # =========================
            # eVTOL side
            # =========================
            evtols = env.vertiports.evtols_at_vertiport.get(str(vid), [])

            charging_evtols = 0
            total_capacity = 0

            remaining_charge_time = []
            remaining_flight_time = []

            for ev in evtols:
                s = ev.state.name

                if s == "CHARGING":
                    charging_evtols += 1

                # ⭐ 最大可用运力（不依赖是否已有乘客）
                total_capacity += ev.spec.capacity

                if hasattr(ev, "remaining_charge_time"):
                    remaining_charge_time.append(ev.remaining_charge_time)

                if hasattr(ev, "remaining_flight_time"):
                    remaining_flight_time.append(ev.remaining_flight_time)

            total_evtols = len(evtols)

            # =========================
            # Time statistics
            # =========================
            avg_charge_time = (
                float(np.mean(remaining_charge_time))
                if remaining_charge_time else 0.0
            )

            # ⭐ 最快可起飞的 eVTOL
            min_charge_time = (
                float(np.min(remaining_charge_time))
                if remaining_charge_time else 0.0
            )

            avg_flight_time = (
                float(np.mean(remaining_flight_time))
                if remaining_flight_time else 0.0
            )

            # =========================
            # Aggregate
            # =========================
            obs.extend([
                float(waiting_cnt),        # 等待乘客数
                float(incoming_cnt),       # 在途乘客数
                float(charging_evtols),    # 充电中的 eVTOL 数
                float(total_evtols),       # eVTOL 总数
                float(total_capacity),     # ⭐ 最大可用运力
                avg_charge_time,           # 平均充电剩余时间
                min_charge_time,           # ⭐ 最短充电时间（最快起飞）
                avg_flight_time,           # 平均剩余飞行时间
            ])

        return np.array(obs, dtype=np.float32)
