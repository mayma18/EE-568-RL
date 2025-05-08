import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from pathlib import Path

def train_model(env_id, total_steps, seed=42):
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    
    # Generate model name based on env_id and total_steps
    model_name = f"ppo_{env_id.lower().replace('-', '_')}_{total_steps // 1000}k"
    
    # Create paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    logs_dir = base_dir / "logs" / model_name
    checkpoint_dir = base_dir / "checkpoints" / model_name
    
    # Ensure directories exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create monitored environment
    env = gym.make(env_id)
    env = Monitor(env, filename=str(logs_dir / model_name))

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=total_steps // 10,
        save_path=str(checkpoint_dir),
        name_prefix=model_name
    )

    print(f"Training {model_name} for {total_steps} steps...")
    model.learn(total_timesteps=total_steps, callback=checkpoint_callback)

    # Save final model
    final_model_path = checkpoint_dir / model_name
    print(f"Training completed. Model saved at: {final_model_path}")
    
    env.close()
    return str(final_model_path)

# Train models with different environments and steps
if __name__ == "__main__":
    path = train_model(env_id="Pendulum-v1", total_steps=600_000)
    print(f"Trained models saved at:\n{path}")
