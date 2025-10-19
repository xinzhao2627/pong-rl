import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import ale_py
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import numpy as np

# TOTAL_TIMESTEPS = 4_000_000
TOTAL_TIMESTEPS = 50000
LOG_INTERVAL = 10
CHECKPOINT_FREQ = 25_000
log_dir = "./sb3_pong_logs/"
save_dir = "./sb3_pong_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=save_dir,
    name_prefix="pong_dqn_model",
    save_replay_buffer=False,
    save_vecnormalize=True,
)
# config = {
#     "env_name": "PongNoFrameskip-v4",
#     "num_envs": 2,
#     "seed": 100,
# }
# env = make_atari_env(config["env_name"], n_envs=config["num_envs"], seed=config["seed"])
env = gym.make("PongNoFrameskip-v4")
# env = VecFrameStack(env, n_stack=4)
print("Environment created and wrapped.")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=100000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log=log_dir,
)


print("\nDQN Model Initialized. Starting training...")


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    log_interval=LOG_INTERVAL,
)

model.save(os.path.join(save_dir, "pong_dqn_final_model"))
print("\nTraining finished. Final model saved.")

env.close()
