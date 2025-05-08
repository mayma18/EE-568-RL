import numpy as np
import gym
from stable_baselines3.common.base_class import BaseAlgorithm

def rollout_policy(
    policy: BaseAlgorithm,
    env_name: str,
    deterministic: bool = True,
    render: bool = False,
    seed: int = 66
):
    """
    Roll out a policy in the given environment for evaluation.

    Args:
        policy (BaseAlgorithm): Trained policy.
        env_name (str): Name of the Gym environment.
        num_episodes (int): Number of episodes to run.
        deterministic (bool): Whether to use deterministic actions.
        render (bool): Whether to render the environment visually.
        seed (int): Random seed for reproducibility.

    Returns:
        avg_reward (float): Average total reward over all episodes.
        trajectories (List[Dict]): Each dict contains 'obs', 'actions', 'rewards'.
    """
    env = gym.make(env_name)
    env.reset(seed=seed)  # Set initial seed
    np.random.seed(seed)

    all_rewards = list()
    trajectories = list()


    obs = env.reset(seed=seed)
    done = False
    total_reward = 0
    ep_obs = list()
    ep_actions = list()
    ep_rewards = list()

    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        ep_obs.append(obs)
        ep_actions.append(action)
        obs, reward, done, _ = env.step(action)
        ep_rewards.append(reward)
        total_reward += reward

        if render:
            env.render()

    all_rewards.append(total_reward)
    trajectories.append({
        "observations": np.array(ep_obs),
        "actions": np.array(ep_actions),
        "rewards": np.array(ep_rewards),
        "total_reward": total_reward
    })

    env.close()
    total_reward = np.sum(all_rewards)
    return total_reward, trajectories
