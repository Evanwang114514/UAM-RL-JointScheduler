from dataclasses import dataclass, field
from typing import Optional
from at_obj.evtol.vehicle_state import VehicleState
from at_obj.evtol.evtol_spec import eVTOLSpec


@dataclass
class eVTOL:
    id: str
    spec: eVTOLSpec

    # 动态状态
    state: VehicleState = VehicleState.IDLE
    current_vertiport_id: Optional[str] = None
    target_vertiport_id: Optional[str] = None

    # 资源状态
    battery_kwh: float = field(init=False)
    # passenger_count: int = 0

    passenger_ids: list = field(default_factory=list)

    # 时间状态（用于仿真）
    remaining_time: float = 0.0  # 当前状态剩余时间（分钟）
    next_available_time: float = 0.0  # 当前航班结束时间

     # ===== 新增（必须）=====
    flight_distance_km: float = 0.0
    planned_energy_kwh: float = 0.0
    just_arrived: bool = False
    just_departed: bool = False
    battery_pct: float = 1.0


    def __post_init__(self):
        self.battery_kwh = self.spec.battery_capacity_kwh
        
    # =======================
    # 状态检查
    # =======================
    def is_available(self) -> bool:
        return self.state in {VehicleState.IDLE, VehicleState.WAITING}

    def can_accept_passengers(self, num: int) -> bool:
         return len(self.passenger_ids) + num <= self.spec.capacity

    def has_sufficient_energy(self, distance_km: float) -> bool:
        required = distance_km * self.spec.energy_consumption_kwh_per_km
        reserve = self.spec.battery_capacity_kwh * self.spec.min_reserve_ratio
        return self.battery_kwh - required >= reserve

    def start_charging(self):
        self.state = VehicleState.CHARGING
        self.remaining_time = 0  # 充电在 step 中累加

    # =======================
    # 仿真时间推进
    # =======================
    def step(self, delta_time_min: float):
        self.just_arrived = False
        self.just_departed = False
        battery_pct = self.battery_kwh / self.spec.battery_capacity_kwh
        self.battery_pct = battery_pct

        if self.state == VehicleState.FLYING:
            self.remaining_time -= delta_time_min
            if self.remaining_time <= 0:
                self._finish_flight()

        elif self.state == VehicleState.CHARGING:
   
            if battery_pct >= 1.0:
                self.state = VehicleState.IDLE
            # ===== 关键修复：防止“没充满就 IDLE” =====
        elif self.state == VehicleState.IDLE:
            if battery_pct < 1.0:
                self.state = VehicleState.CHARGING

            
    def start_flight(
        self,
        dest_vertiport_id: str,
        distance_km: float,
        passenger_ids: list
    ):
        assert passenger_ids, f"{self.id} must have passengers to start flight!"
        assert self.state == VehicleState.IDLE, "EVTOL must be IDLE to start flight"
        assert len(passenger_ids) <= self.spec.capacity

        self.target_vertiport_id = dest_vertiport_id
        self.passenger_ids = list(passenger_ids)

        # 飞行时间（分钟）
        flight_time_min = distance_km / self.spec.max_speed * 60.0
        self.remaining_time = flight_time_min

        # 能量规划
        self.flight_distance_km = distance_km
        self.planned_energy_kwh = (
            distance_km * self.spec.energy_consumption_kwh_per_km
        )

        # 扣电（起飞即扣）
        self.battery_kwh -= self.planned_energy_kwh

        self.state = VehicleState.FLYING

    
    def _finish_flight(self):
        self.state = VehicleState.CHARGING
        self.current_vertiport_id = self.target_vertiport_id

        self.target_vertiport_id = None
        self.flight_distance_km = 0.0
        self.remaining_time = 0.0

        arrived_passengers = list(self.passenger_ids)  # 另一种复制方法
        # self.passenger_ids.clear() #会导致 arrived_passengers 也被清空
        self.just_arrived = True
        return arrived_passengers

    # def _finish_current_state(self, delta_time_min: float):
    #     if self.state == VehicleState.FLYING:
    #         self.state = VehicleState.LANDING
    #         self.remaining_time = 1  # 假设降落耗 1 分钟

    #     elif self.state == VehicleState.LANDING:
    #         self.current_vertiport_id = self.target_vertiport_id
    #         self.target_vertiport_id = None
    #         self.passenger_count = 0
    #         self.just_arrived = True # 标记到达
    #         self.state = VehicleState.CHARGING
    #         self.remaining_time = 0  # 充电在 step 累加

    #     elif self.state == VehicleState.CHARGING:
    #         self.battery_kwh = min(self.battery_kwh + delta_time_min * self.spec.charge_rate_kwh_per_min,
    #                                self.spec.battery_capacity_kwh)
    #         if self.battery_kwh >= self.spec.battery_capacity_kwh:
    #             self.state = VehicleState.IDLE

    # =======================
    # 查询接口
    # =======================
    def get_status(self):
        return {
            "id": self.id,
            "state": self.state,
            "current_vertiport": self.current_vertiport_id,
            "target_vertiport": self.target_vertiport_id,
            "battery_pct": self.battery_pct,
            "passengers": self.passenger_ids,
            "remaining_time": self.remaining_time
        }
