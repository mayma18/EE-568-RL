# test_equivalence.py

import torch
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
from policy.shit.stable_to_net import ActorCriticNetwork
import gymnasium as gym

def test_equivalence(env_id, model_path, actor_state_dict_path, num_tests=100, tol=1e-5):
    # Load SB3 PPO model
    sb3_model = PPO.load(model_path, device="cpu")
    sb3_policy = sb3_model.policy

    # Create extracted PyTorch actor
    actor = ActorCriticNetwork(sb3_policy)
    actor.load_state_dict(torch.load(actor_state_dict_path, map_location="cpu"))
    actor.eval()

    # Create environment
    env = gym.make(env_id)
    
    match_count = 0
    for i in range(num_tests):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Get SB3 action (deterministic)
        sb3_action, _ = sb3_policy.predict(obs, deterministic=True)

        # Get action from extracted PyTorch actor
        with torch.no_grad():
            action_tensor = actor.predict(obs_tensor, deterministic=True)
            actor_action = action_tensor.squeeze(0).cpu().numpy()

        # Compare
        if np.allclose(sb3_action, actor_action, atol=tol):
            match_count += 1
        else:
            print(f"[Mismatch] Step {i+1}")
            print(f"SB3 Action   : {sb3_action}")
            print(f"Actor Action : {actor_action}")

    print(f"\nTest finished: {match_count}/{num_tests} actions matched within tolerance {tol}")

if __name__ == "__main__":
    env_id = "Pendulum-v1"
    total_steps = 200_000
    model_name = f"ppo_{env_id.lower().replace('-', '_')}_{total_steps // 1000}k"
    model_path = Path("checkpoints") / model_name / f"{model_name}_{total_steps}_steps"
    actorcritic_path = Path("checkpoints") / model_name / f"{model_name}_net.pt"

    test_equivalence(env_id, str(model_path), str(actorcritic_path))
