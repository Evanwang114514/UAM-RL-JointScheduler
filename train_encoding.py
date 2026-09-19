'''
Author: pangay 1623253042@qq.com
Date: 2024-01-11 20:58:46
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2025-04-29 16:01:24
'''
    
import os
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from utilss.feature_extraction import CustomModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from utilss.sb3_utils import VecNormalizeCallback, linear_schedule
from utilss.make_env import make_env
from utilss.encoding import Embedding
path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), terminal_log_level="INFO")

if __name__ == '__main__':
    log_path = path_convert('./log/')
    model_path = path_convert('./model_encode_state_5/')
    tensorboard_path = path_convert('./tensorboard/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    
    # #########
    # Init Env
    # #########
    
    params = {
        'log_file':log_path,
    }
    env =SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(5)]) 
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    # #########
    # Callback
    # #########
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # 多少个 step, 需要根据与环境的交互来决定
        save_path=model_path,
    )
    vec_normalize_callback = VecNormalizeCallback(
        save_freq=10000,
        save_path=model_path,
    ) # 保存环境参数
    callback_list = CallbackList([checkpoint_callback, vec_normalize_callback])

    # #########
    # Training
    # #########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_kwargs = dict(
       features_extractor_class=Embedding,
       features_extractor_kwargs=dict(features_dim = 16*5),
    )
    model = PPO(
                "MlpPolicy", 
                env, 
                batch_size=64,
                n_steps = 3000, n_epochs=1, # 每次间隔 n_epoch 去评估一次
                learning_rate=linear_schedule(5e-4),
                verbose=True, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_path, 
                device=device
            )
    model.learn(total_timesteps=1e6, callback=callback_list)
    
    # #################
    # 保存 model 和 env
    # #################
    env.save(f'{model_path}/last_vec_normalize.pkl')
    model.save(f'{model_path}/last_rl_model.zip')
    print('训练结束, 达到最大步数.')

    env.close()