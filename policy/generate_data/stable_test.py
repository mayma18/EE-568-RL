# test.py

import gymnasium as gym
from pathlib import Path
import torch
from stable_baselines3 import PPO
import numpy as np
import random

def test_model(env_id, model_path, render=True, max_steps=500, seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Construct model path
    model_name = f"ppo_{env_id.lower().replace('-', '_')}_{total_steps // 1000}k"
    model_path = Path("checkpoints") / model_name / f"{model_name}_{total_steps}_steps"

    # Load trained model
    model = PPO.load(str(model_path))

    # Create evaluation environment
    env = gym.make(env_id, render_mode="human" if render else None)
    obs, _ = env.reset(seed=seed)

    # Run the policy
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if render:
            env.render()
        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    env_id = "Pendulum-v1"
    total_steps = 200_000
    test_model(env_id, total_steps)
