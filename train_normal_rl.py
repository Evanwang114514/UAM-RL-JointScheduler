'''
Author: pangay 1623253042@qq.com
Date: 2025-05-13
FilePath: /UAGMC/train_rl.py
Description: PPO training for UAM RL environment
'''

# =========================
# Imports
# =========================
import torch
from pathlib import Path
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import TimeLimit

from utilss.make_env import make_env
from utilss.sb3_utils import linear_schedule, VecNormalizeCallback


# =========================
# Path utils
# =========================
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

LOG_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)


# =========================
# Main
# =========================
if __name__ == "__main__":

    # -------------------------
    # 1. Environment config
    # -------------------------
    n_envs = 1
    max_time = 420

    params = {
    "log_dir": LOG_DIR,
    "candidate_from_vertiports": [0, 1],
    "to_vertiport": 2,
    }

    env = DummyVecEnv([
        make_env(
            max_time=max_time,
            log_dir=LOG_DIR,
            env_index=i,
            candidate_from_vertiports = [0, 1],
            to_vertiport = 2,
            person_spawn_file= "passengers.csv",   #"passengers.csv"  # 固定 or None
            enable_logger = False,
                            )
        for i in range(n_envs)
    ])
    #env = TimeLimit(env, max_episode_steps = 420)
    # -------------------------
    # 2. VecNormalize
    # -------------------------
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )

    # -------------------------
    # 3. PPO config
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(5e-4),
        n_steps=4096,
        batch_size= 512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=str(LOG_DIR / "tb")
    )

    # -------------------------
    # 4. Callbacks
    # -------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="uam_ppo"
    )

    vecnorm_callback = VecNormalizeCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="vecnorm"
    )

    # -------------------------
    # 5. Training
    # -------------------------
    total_timesteps = 2_000_0000

    logger.info("Start training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, vecnorm_callback]
    )

    # -------------------------
    # 6. Save final model
    # -------------------------
    model.save(MODEL_DIR / "final_rl_model")
    env.save(MODEL_DIR / "final_vec_normalize.pkl")

    env.close()
    logger.success("Training finished.")
