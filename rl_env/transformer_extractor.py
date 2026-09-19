import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


# =========================
# Temporal Feature Extractor
# =========================
class TemporalTransformerExtractor(BaseFeaturesExtractor):
    """
    RL-friendly temporal feature extractor for ablation studies.

    Ablation dimensions:
      - Encoder: on / off
      - Temporal model: none / GRU / Transformer
      - Aggregation: mean / last
    """

    def __init__(
        self,
        observation_space,
        features_dim=128,

        # ===== encoder =====
        use_encoder=True,

        # ===== temporal =====
        temporal_type="transformer",  # "none" | "gru" | "transformer"

        # GRU
        gru_hidden_dim=128,

        # Transformer
        n_layers=1,
        n_heads=2,
        ff_dim=256,

        # ===== aggregation =====
        aggregation="mean",  # "mean" | "last"
    ):
        super().__init__(observation_space, features_dim)

        T, obs_dim = observation_space.shape
        self.T = T
        self.features_dim = features_dim

        self.use_encoder = use_encoder
        self.temporal_type = temporal_type
        self.aggregation = aggregation

        # =====================
        # 1. Frame Encoder
        # =====================
        if use_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, features_dim),
                nn.LayerNorm(features_dim),
                nn.ReLU(),
            )
        else:
            self.encoder = nn.Identity()
            self.input_proj = nn.Linear(obs_dim, features_dim)

        # =====================
        # 2. Temporal Model
        # =====================
        if temporal_type == "none":
            self.temporal_model = nn.Identity()

        elif temporal_type == "gru":
            self.temporal_model = nn.GRU(
                input_size=features_dim,
                hidden_size=features_dim,
                batch_first=True,
            )

        elif temporal_type == "transformer":
            self.pos_encoder = PositionalEncoding(features_dim, max_len=T + 1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=features_dim,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_model = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
            )

        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        # =====================
        # 3. Output Head
        # =====================
        self.output_proj = nn.Linear(features_dim, features_dim)

    # =====================
    # Forward
    # =====================
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, obs_dim)
        """
        # ----- Encoder -----
        if self.use_encoder:
            x = self.encoder(obs)  # (B, T, D)
        else:
            x = self.input_proj(obs)

        # ----- Temporal modeling -----
        if self.temporal_type == "gru":
            x, _ = self.temporal_model(x)

        elif self.temporal_type == "transformer":
            x = self.pos_encoder(x)
            x = self.temporal_model(x)

        # ----- Aggregation -----
        if self.aggregation == "mean":
            h = x.mean(dim=1)
        elif self.aggregation == "last":
            h = x[:, -1]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self.output_proj(h)
