import gymnasium as gym
from pathlib import Path
import torch
from stable_baselines3 import PPO
import numpy as np
import random
from tqdm import tqdm
import pickle
import copy

def generate_dpo_dataset(
    env_id: str,
    model_name_pi1: str,
    model_name_pi2: str,
    total_steps_pi1: int,
    total_steps_pi2: int,
    rollout_policy_name: str,
    num_trajectories: int = 1000,
    max_steps: int = 500,
    render: bool = False
):
    # Load both policies
    path_pi1 = Path("checkpoints") / model_name_pi1 / f"{model_name_pi1}_{total_steps_pi1}_steps"
    path_pi2 = Path("checkpoints") / model_name_pi2 / f"{model_name_pi2}_{total_steps_pi2}_steps"
    policy1 = PPO.load(str(path_pi1))
    policy2 = PPO.load(str(path_pi2))

    # Choose rollout policy (who is used to step the environment)
    if rollout_policy_name == "pi1":
        rollout_policy = policy1
    elif rollout_policy_name == "pi2":
        rollout_policy = policy2
    elif rollout_policy_name == "random":
        rollout_policy = None
    else:
        raise ValueError("rollout_policy_name must be 'pi1', 'pi2' or 'random'.")

    dpo_data = list()

    for seed in tqdm(range(num_trajectories), desc="Collecting DPO data"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        env = gym.make(env_id, render_mode="human" if render else None)
        obs, _ = env.reset(seed=seed)

        for _ in range(max_steps):
            # Two policies output actions on the same state
            action1, _ = policy1.predict(obs, deterministic=True)
            action2, _ = policy2.predict(obs, deterministic=True)

            env1 = copy.deepcopy(env)
            _, r1, _, _, _ = env1.step(action1)

            env2 = copy.deepcopy(env)
            _, r2, _, _, _ = env2.step(action2)

            # Assign a⁺ and a⁻ based on reward, or use soft stochastic label
            if r1 > r2:
                a_pref, a_rej = action1, action2
                r_pref, r_rej = r1, r2
            else:
                a_pref, a_rej = action2, action1
                r_pref, r_rej = r2, r1

            dpo_data.append({
                "obs": np.array(obs, dtype=np.float32),
                "action_pref": np.array(a_pref, dtype=np.float32),
                "action_rej": np.array(a_rej, dtype=np.float32),
                "reward_pref": float(r_pref),
                "reward_rej": float(r_rej),
                "seed": seed
            })

            # Step env using rollout policy or random
            if rollout_policy is None:
                step_action = env.action_space.sample()
            else:
                step_action, _ = rollout_policy.predict(obs, deterministic=True)

            obs, _, terminated, truncated, _ = env.step(step_action)
            if terminated or truncated:
                break

        env.close()

    # Save to .pkl
    save_name = f"dpo_dataset_{model_name_pi1}_vs_{model_name_pi2}_rollout_{rollout_policy_name}.pkl"
    save_path = Path("data") / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(dpo_data, f)

    print(f"Saved {len(dpo_data)} DPO (s, a⁺, a⁻) samples to {save_path}")


if __name__ == "__main__":
    env_id = "Pendulum-v1"

    generate_dpo_dataset(
        env_id=env_id,
        model_name_pi1="ppo_pendulum_v1_400k",  # better policy
        model_name_pi2="ppo_pendulum_v1_200k",  # worse policy
        total_steps_pi1=400_000,
        total_steps_pi2=200_000,
        rollout_policy_name="pi1",              # rollout policy can be "pi1", "pi2", or "random"
        num_trajectories=1000,
        max_steps=500
    )
