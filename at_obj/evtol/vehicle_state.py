from enum import Enum, auto

class VehicleState(Enum):
    IDLE = auto()        # 空闲，可被调度
    WAITING = auto()     # 等待乘客 / 任务
    BOARDING = auto()    # 登机中
    FLYING = auto()      # 飞行中
    LANDING = auto()     # 降落中
    CHARGING = auto()    # 充电中
    MAINTENANCE = auto() # 维护/不可用