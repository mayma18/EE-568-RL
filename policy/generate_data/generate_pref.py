import pickle
from typing import List, Dict
from pathlib import Path

def load_trajectories_pkl(file_path: str) -> List[Dict]:
    """Load a list of trajectory dictionaries from a pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def compute_total_rewards_from_episodes(episodes: List[Dict]) -> List[float]:
    """Compute total reward for each episode."""
    return [float(sum(ep["rewards"])) for ep in episodes]

def build_preference_dataset(path_pi1, path_pi2, save_path, episode=1000):
    # Load episode lists for both policies
    episodes_pi1 = load_trajectories_pkl(path_pi1)
    episodes_pi2 = load_trajectories_pkl(path_pi2)

    # Compute total rewards
    rewards_pi1 = compute_total_rewards_from_episodes(episodes_pi1)
    rewards_pi2 = compute_total_rewards_from_episodes(episodes_pi2)

    preference_pairs = list()

    for idx in range(episode):

        ep1 = episodes_pi1[idx]
        ep2 = episodes_pi2[idx]
        rew1 = rewards_pi1[idx]
        rew2 = rewards_pi2[idx]

        # If rew1 > rew2, then ep1 is preferred
        preferred = 0 if rew1 > rew2 else 1
        print(f"idx:{idx}, preferred:{preferred}")
        if preferred == 0:
            obs_pref, act_pref, rew_pref = ep1["observations"], ep1["actions"], rew1
            obs_rej, act_rej, rew_rej = ep2["observations"], ep2["actions"], rew2
        else:
            obs_pref, act_pref, rew_pref = ep2["observations"], ep2["actions"], rew2
            obs_rej, act_rej, rew_rej = ep1["observations"], ep1["actions"], rew1

        pair = {
            "obs_pref": obs_pref,
            "act_pref": act_pref,
            "rew_pref": rew_pref,
            "obs_rej": obs_rej,
            "act_rej": act_rej,
            "rew_rej": rew_rej,
        }

        preference_pairs.append(pair)

    # Save as pickle file
    with open(save_path, "wb") as f:
        pickle.dump(preference_pairs, f)

    print(f"Saved {len(preference_pairs)} preference pairs to {save_path}")

if __name__ == "__main__":
    env_id = "Pendulum-v1"
    total_steps_list = [400_000, 200_000,]
    model_names = [
        f"ppo_{env_id.lower().replace('-', '_')}_{steps // 1000}k"
        for steps in total_steps_list
    ]

    pi1_name = model_names[0]
    pi2_name = model_names[1]
    base_dir = Path(__file__).resolve().parent.parent.parent
    path_pi1 = base_dir / f"data/trajectories_{pi1_name}.pkl"
    path_pi2 = base_dir / f"data/trajectories_{pi2_name}.pkl"
    save_path = base_dir / f"data/preference_pairs_{pi1_name}_vs_{pi2_name}.pkl"

    build_preference_dataset(path_pi1, path_pi2, save_path, episode=5000)
