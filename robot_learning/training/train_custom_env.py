import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env

env_id = 'CustomEnv-v0'
save_dir = "./trained_models"
log_dir = "./logs"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

vec_env = make_vec_env(env_id, n_envs=1)

action_noise = NormalActionNoise(mean=0., sigma=0.1)
model = SAC(MlpPolicy, vec_env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir)


checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_dir,
                                         name_prefix='sac_model', verbose=1)

model.learn(total_timesteps=5e6, callback=[checkpoint_callback])

model.save(os.path.join(save_dir, "final_sac_custom__env"))

