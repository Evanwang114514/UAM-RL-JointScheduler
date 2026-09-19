"""
Author: pangay 1623253042@qq.com
Date: 2025-05-13
FilePath: /UAGMC/test_rl.py
Description: PPO evaluation for UAM RL environment
"""

# =========================
# Imports
# =========================
import torch
from pathlib import Path
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utilss.make_env import make_env


# =========================
# Path utils
# =========================
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"


# =========================
# Main
# =========================
if __name__ == "__main__":

    # -------------------------
    # 1. Environment config
    # -------------------------
    n_envs = 1
    max_time = 600

    env = DummyVecEnv([
        make_env(
            max_time=max_time,
            log_dir=LOG_DIR,
            env_index=0,
            candidate_from_vertiports=[0, 1],
            to_vertiport=2,
            person_spawn_file="train_data/passengers_300.csv",  # 与训练保持一致
            enable_logger = True,
        )
    ])

    # -------------------------
    # 2. Load VecNormalize
    # -------------------------
    vecnorm_path = MODEL_DIR / "final_vec_normalize.pkl"
    assert vecnorm_path.exists(), "VecNormalize file not found!"

    env = VecNormalize.load(vecnorm_path, env)

    # ⚠️ 测试时一定要关
    env.training = False
    env.norm_reward = False

    # -------------------------
    # 3. Load model
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = MODEL_DIR / "final_rl_model.zip"
    assert model_path.exists(), "RL model not found!"

    model = PPO.load(
        model_path,
        env=env,
        device=device
    )

    logger.info("Model and environment loaded. Start evaluation.")

    # -------------------------
    # 4. Run evaluation
    # -------------------------
    obs = env.reset()
    done = False

    episode_reward = 0.0
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        episode_reward += reward[0]
        step_count += 1

    # -------------------------
    # 5. Results
    # -------------------------
    logger.success(f"Evaluation finished.")
    logger.info(f"Total steps: {step_count}")
    logger.info(f"Episode reward: {episode_reward:.3f}")

    env.close()
