import gymnasium as gym
from pathlib import Path
import torch
from stable_baselines3 import PPO
import numpy as np
import random
from tqdm import tqdm
import pickle
from stable_baselines3.common.utils import set_random_seed

def generate_trajectory(
    env_id: str,
    model_path: str,
    episode: int = 1000,
    seed: int=42,
    render: bool = False
):
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    # load model
    model = PPO.load(str(model_path))

    all_episodes = list()

    for idx in tqdm(range(episode), desc=f"Generating for {model_name}"):

        env = gym.make(env_id, render_mode="human" if render else None)
        max_steps = env.spec.max_episode_steps
        obs, _ = env.reset(seed=idx)

        obs_traj = list()
        act_traj = list()
        rew_traj = list()

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            obs_traj.append(obs)
            act_traj.append(action)
            rew_traj.append(reward)

        env.close()

        episode = {
            "observations": np.array(obs_traj),
            "actions": np.array(act_traj),
            "rewards": np.array(rew_traj),
            "seed": idx
        }
        all_episodes.append(episode)

    # save model
    save_path = base_dir / f"data/trajectories_{model_name}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(all_episodes, f)

    print(f"Saved {len(all_episodes)} episodes to {save_path}")

if __name__ == "__main__":
    env_id = "Pendulum-v1"
    total_steps_list = [400_000, 200_000,]

    base_dir = Path(__file__).resolve().parent.parent.parent
    for tatal_steps in total_steps_list:
        model_name = f"ppo_{env_id.lower().replace('-', '_')}_{tatal_steps // 1000}k"
        model_path = base_dir / "checkpoints" / model_name / f"{model_name}_{tatal_steps}_steps"
        generate_trajectory(env_id, model_path, episode=5000)
