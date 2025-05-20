def load_model(model_path):
    from stable_baselines3 import PPO
    return PPO.load(model_path)

def read_model_files(models_directory):
    import os
    return [os.path.join(models_directory, f) for f in os.listdir(models_directory) if f.endswith('.zip') or f.endswith('.pkl')]

def format_results(results):
    import numpy as np
    average = np.mean(results, axis=0)
    variance = np.var(results, axis=0)
    return average, variance

def process_simulation_data(simulation_data):
    total_rewards = [data['total_reward'] for data in simulation_data]
    displacements = [data['displacement'] for data in simulation_data]
    angles = [data['angle'] for data in simulation_data]
    return total_rewards, displacements, angles