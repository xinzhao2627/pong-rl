import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
    VecMonitor,
)
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import ale_py
import json
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))


class CustomCnnPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[],
            **kwargs,
        )


TOTAL_TIMESTEPS = 5_000_000
LOG_INTERVAL = 10
CHECKPOINT_FREQ = 25_000
log_dir = "./sb3_pong_logs_ppo/"
save_dir = "./sb3_pong_models_ppo/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=save_dir,
    name_prefix="pong_ppo_model",
    save_replay_buffer=False,
    save_vecnormalize=True,
)
rewards = []
lengths = []


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_num = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.episode_num += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                rewards.append(ep_reward)
                lengths.append(ep_length)
        return True


config = {
    "env_name": "PongNoFrameskip-v4",
    "num_envs": 8,
    "seed": 100,
}
env = make_atari_env(config["env_name"], n_envs=config["num_envs"], seed=config["seed"])
env = VecFrameStack(env, n_stack=4)
print("Environment created and wrapped.")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
model = PPO(
    CustomCnnPolicy,
    env,
    batch_size=32,
    learning_rate=1e-4,
    clip_range=0.1,
    ent_coef=0.01,
    gae_lambda=0.9,
    max_grad_norm=0.5,
    n_epochs=4,
    n_steps=128,
    vf_coef=0.5,
    tensorboard_log=log_dir,
    verbose=1,
)
print(model.policy)

print("\nppo Model Initialized. Starting training...")

callback_list = CallbackList([checkpoint_callback, RewardLoggingCallback()])


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callback_list,
    log_interval=LOG_INTERVAL,
)


model.save(os.path.join(save_dir, "pong_ppo_final_model"))
data_dict = {"episode_rewards": rewards, "episode_lengths": lengths}
with open(f"ppo_res.json", "w") as json_file:
    json.dump(data_dict, json_file)

print("\nTraining finished. Final model saved.")
env.close()
