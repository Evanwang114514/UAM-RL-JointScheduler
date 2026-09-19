from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """
    所有策略的统一接口
    """

    @abstractmethod
    def act(self, env, state):
        """
        输入当前 env 和 state
        输出 actions: dict[pid -> action]
        """
        pass
