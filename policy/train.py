# train_sb3_policy.py

import torch
import gym
import random
import numpy as np
import wandb
from pathlib import Path
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Type
from stable_baselines3.common.base_class import BaseAlgorithm

# Supported algorithms
SB3_ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG
}

def train_sb3_policy(
    env_id: str,
    algo: str,
    total_timesteps: int,
    save_path: str,
    seed: int = 42,
    log_wandb: bool = True,
    wandb_project: str = "dpo-training"
):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # Resolve save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if log_wandb:
        wandb.init(
            project=wandb_project,
            name=f"{algo}_{env_id}_{total_timesteps // 1000}k",
            config={
                "algo": algo,
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                "seed": seed,
            }
        )

    # Create environment with monitoring
    env = gym.make(env_id)
    env = Monitor(env)

    # Select algorithm class
    algo_cls: Type[BaseAlgorithm] = SB3_ALGO_MAP[algo.lower()]
    
    # Create policy
    model = algo_cls("MlpPolicy", env, verbose=1, seed=seed)

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=total_timesteps // 5,
        save_path=str(save_path),
        name_prefix=f"{algo}_{env_id}"
    )

    print(f"Training {algo.upper()} on {env_id} for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    final_model_path = save_path / f"{algo}_{env_id}_final.zip"
    model.save(final_model_path)

    print(f"Model saved at: {final_model_path}")

    if log_wandb:
        wandb.finish()

    env.close()
    return str(final_model_path)
