import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# 没有 attention encoder 稳定  # 补充实验
class TemporalLSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM-based temporal feature extractor for ablation study.
    Aligned with TemporalTransformerExtractor.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,

        # ===== encoder =====
        use_encoder: bool = True,

        # ===== LSTM =====
        hidden_dim: int = 128,
        num_layers: int = 1,

        # ===== aggregation =====
        aggregation: str = "last",  # "last" | "mean"
    ):
        super().__init__(observation_space, features_dim)

        T, obs_dim = observation_space.shape
        self.T = T
        self.aggregation = aggregation
        self.use_encoder = use_encoder

        # =====================
        # 1. Frame Encoder
        # =====================
        if use_encoder:
            self.frame_encoder = nn.Linear(obs_dim, features_dim)
            lstm_input_dim = features_dim
        else:
            self.frame_encoder = nn.Identity()
            self.input_proj = nn.Linear(obs_dim, features_dim)
            lstm_input_dim = features_dim

        # =====================
        # 2. Temporal Model (LSTM)
        # =====================
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # =====================
        # 3. Output head
        # =====================
        self.output_proj = nn.Linear(hidden_dim, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, obs_dim)
        """
        B, T, _ = obs.shape

        # ----- Frame encoding -----
        if self.use_encoder:
            x = self.frame_encoder(obs)      # (B, T, D)
        else:
            x = self.input_proj(obs)          # (B, T, D)

        # ----- Temporal modeling -----
        lstm_out, _ = self.lstm(x)            # (B, T, H)

        # ----- Aggregation -----
        if self.aggregation == "last":
            h = lstm_out[:, -1]               # (B, H)
        elif self.aggregation == "mean":
            h = lstm_out.mean(dim=1)          # (B, H)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self.output_proj(h)
