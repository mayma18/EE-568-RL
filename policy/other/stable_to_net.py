# extract_policy.py

import torch
from stable_baselines3 import PPO
from pathlib import Path
import torch
from policy.shit.actor_critic import ActorCriticNetwork

def export_policy(model_path, save_path):
    model = PPO.load(model_path, device="cpu")
    actor = ActorCriticNetwork(model.policy)
    actor.eval()
    torch.save(actor.state_dict(), save_path)
    print(f"Saved pure PyTorch actor network to {save_path}")

if __name__ == "__main__":
    env_id = "Pendulum-v1"
    total_steps = 200_000
    model_name = f"ppo_{env_id.lower().replace('-', '_')}_{total_steps // 1000}k"
    model_path = Path("checkpoints") / model_name / f"{model_name}_{total_steps}_steps"
    save_path = Path("checkpoints") / model_name / f"{model_name}_net.pt"
    export_policy(str(model_path), str(save_path))
