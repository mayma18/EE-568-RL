import os
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class ModelEvaluator:
    def __init__(self, model_paths, env):
        self.model_paths = model_paths
        self.env = env

    def load_models(self):
        from stable_baselines3 import PPO
        models = {}
        for path in self.model_paths:
            model_name = path.split(os.sep)[-1]
            models[model_name] = PPO.load(path, device="cpu")
        return models

    def run_simulations(self, model, num_episodes=100):
        total_rewards = []
        displacements = []
        angles = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()  # <-- unpack obs, info
            done = False
            total_reward = 0
            total_displacement = 0
            total_angle = 0

            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                # obs: [cart position, cart velocity, pole angle, pole velocity at tip]
                total_displacement += abs(obs[0])
                total_angle += abs(obs[2])

            total_rewards.append(total_reward)
            displacements.append(total_displacement)
            angles.append(total_angle)

        return {
            'average_reward': sum(total_rewards) / num_episodes,
            'variance_reward': sum((x - (sum(total_rewards) / num_episodes)) ** 2 for x in total_rewards) / num_episodes,
            'average_displacement': sum(displacements) / num_episodes,
            'variance_displacement': sum((x - (sum(displacements) / num_episodes)) ** 2 for x in displacements) / num_episodes,
            'average_angle': sum(angles) / num_episodes,
            'variance_angle': sum((x - (sum(angles) / num_episodes)) ** 2 for x in angles) / num_episodes,
        }