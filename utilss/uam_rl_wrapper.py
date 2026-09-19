import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque   # NEW

from rl_env.observation_encoder import ObservationEncoder
from rl_env.action_decoder import ActionDecoder


class UAMRLWrapper(gym.Env):
    """
    RL wrapper on top of Scenario.
    ALL RL logic stays here.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        scenario,
        candidate_from_vertiports,
        to_vertiport,
        max_time: int = 420,
        num_frames: int = 6,   # NEW: 帧堆叠数
    ):
        super().__init__()

        self.env = scenario
        self.max_time = max_time
        self.num_frames = num_frames  # NEW

        self.encoder = ObservationEncoder(
            num_vertiports=(len(self.env.vertiports.vertiport_list) - 1) # NEW: 排除目标vertiport
        )

        self.decoder = ActionDecoder(
            candidate_from_vertiports,
            to_vertiport
        )

        # ===== Frame buffer =====
        self.obs_buffer = deque(maxlen=num_frames)  # NEW

        # ===== Gym spaces =====
        self.single_obs_dim = self.encoder.obs_dim  # NEW

        self.observation_space = spaces.Box(
            low=0,
            high=1e6,
            shape=(self.single_obs_dim * num_frames,),  # NEW
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(
            len(candidate_from_vertiports)
        )

        self.state = None

    # =====================
    # Utils
    # =====================
    def _get_stacked_obs(self):
        """
        Concatenate stacked frames into one vector
        """
        return np.concatenate(list(self.obs_buffer), axis=0).astype(np.float32)

    # =====================
    # Gym API
    # =====================
    def reset(self, *, seed=None, options=None):
        self.state = self.env.reset()

        obs = self.encoder.encode(self.env, self.state)

        # ===== init frame stack =====
        self.obs_buffer.clear()
        for _ in range(self.num_frames):
            self.obs_buffer.append(obs)   # 重复第一帧

        return self._get_stacked_obs(), {}

    def step(self, action: int):

        # =========================
        # 1. Prepare action for env
        # =========================
        waiting = self.state["waiting_decisions"]

        env_action = None
        if len(waiting) > 0:
            pid = waiting[0]
            env_action = self.decoder.decode(action, pid)

        # =========================
        # 2. Single env step
        # =========================
        self.state, env_reward, terminated, truncated, info = self.env.step(env_action)

        # =========================
        # 3. Observation
        # =========================
        obs = self.encoder.encode(self.env, self.state)

        # ===== update frame stack =====
        self.obs_buffer.append(obs)

        stacked_obs = self._get_stacked_obs()

        return stacked_obs, env_reward, terminated, truncated, info
