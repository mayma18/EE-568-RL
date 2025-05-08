# test_exported_actor_critic.py

import gymnasium as gym
from pathlib import Path
import torch
from policy.other.actor_critic import ActorCriticNetwork
from stable_baselines3 import PPO
import numpy as np
import random

def test_model(env_id, model_path, render=True, max_steps=500, seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Load dummy SB3 model to construct network architecture
    dummy_model = PPO("MlpPolicy", gym.make(env_id), device="cpu") 
    actor_critic = ActorCriticNetwork(dummy_model.policy)
    actor_critic.load_state_dict(torch.load(model_path, map_location="cpu"))
    actor_critic.eval()
    device = next(actor_critic.parameters()).device
    print("Device:", device)

    # Create evaluation environment
    env = gym.make(env_id, render_mode="human" if render else None)
    obs, _ = env.reset(seed=seed)

    for step in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = actor_critic.predict(obs_tensor)

        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action_np)

        if render:
            env.render()

        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    env_id = "Pendulum-v1"
    total_steps = 200_000
    model_name = f"ppo_{env_id.lower().replace('-', '_')}_{total_steps // 1000}k"
    model_path = Path("checkpoints") / model_name / f"{model_name}_net.pt"
    test_model(env_id, model_path)
