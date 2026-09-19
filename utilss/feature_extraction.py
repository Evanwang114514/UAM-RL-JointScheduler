import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 16):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        net_shape = 36

        self.embedding = nn.Sequential(
            nn.Linear(net_shape, 32),
            nn.ReLU(),
        ) # 5* -> 5*32
        
        self.lstm = nn.LSTM(
            input_size=32, hidden_size=64,
            num_layers=1, batch_first=True
        )
        self.relu = nn.ReLU()

        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, features_dim)
        )

    def forward(self, observations):
        
        batch_size = observations.shape[0]
        time_length =  observations.shape[1]
        observations = observations.reshape(batch_size * time_length,-1)
        embedding = self.embedding(observations)
        embedding = embedding.reshape(batch_size,time_length,-1)
        output, (hn, cn) = self.lstm(embedding)
        hn = hn.view(-1, 64)
        hn = self.relu(hn)
        
        output = self.output(hn)
        return output