import torch
from torch.utils.data import DataLoader, random_split
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from dataset.dpo_dataset import DpoTrajectoryDataset, PrefDataset, collate_fn
from loss.dpo_loss import TrajectoryScorer, DPOTrajectoryLoss
from env_runner.simple_env_runner import rollout_policy
from pathlib import Path
import numpy as np
import pandas as pd
import ast
import random
import wandb  # Weights & Biases for logging
from tqdm import tqdm  # Progress bar
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from typing import Type
from stable_baselines3.common.base_class import BaseAlgorithm


SB3_ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG
}

def align_with_dpo_pipeline(
    env_id: str,
    dpo_dataset_path: str,
    policy_path: str,
    ref_policy_path: str,
    save_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-5,
    beta: float = 1.0,
    seed: int = 42,
    val_ratio: float = 0.1,
    save_model: int = 10,
    roll_out: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    policy_algo: str = "ppo",
    ref_policy_algo: str = "ppo"
):
    # Set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # Initialize Weights & Biases logging
    wandb.init(
        project="dpo-training",  # Change to your project name if needed
        name=f"dpo_{Path(policy_path).stem}_vs_{Path(ref_policy_path).stem}",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "beta": beta,
            "seed": seed,
        }
    )

    # Convert all paths to Path objects
    dpo_dataset_path = Path(dpo_dataset_path)
    policy_path = Path(policy_path)
    ref_policy_path = Path(ref_policy_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and split into train/validation
    
    df = pd.read_csv(dpo_dataset_path)
    full_dataset = PrefDataset(df)
    
    #full_dataset = DpoTrajectoryDataset(dpo_dataset_path)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,collate_fn=collate_fn )

    # Load policies not necessary PPO 
    
    train_algo_cls: Type[BaseAlgorithm] = SB3_ALGO_MAP[policy_algo.lower()]
    ref_algo_cls: Type[BaseAlgorithm] = SB3_ALGO_MAP[ref_policy_algo.lower()]

    train_policy = train_algo_cls.load(str(policy_path))
    ref_policy = ref_algo_cls.load(str(ref_policy_path))

    

    train_policy.policy.to(device)
    ref_policy.policy.to(device)
    ref_policy.policy.eval()

    # Set up scorers and DPO loss
    pi_scorer = TrajectoryScorer(train_policy.policy)
    ref_scorer = TrajectoryScorer(ref_policy.policy)
    loss_fn = DPOTrajectoryLoss(beta=beta)
    optimizer = torch.optim.Adam(train_policy.policy.parameters(), lr=lr)

    train_losses = list()
    val_losses = list()
    
    for epoch in range(epochs):
        train_policy.policy.train()
        epoch_train_loss = 0.0

        # Training loop with tqdm
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch in train_loader_tqdm:
        
            obs_pref = batch["obs_w"].to(device)
            act_pref = batch["act_w"].to(device)
            obs_rej = batch["obs_l"].to(device)
            act_rej = batch["act_l"].to(device)

            log_pi_pref = pi_scorer(obs_pref, act_pref)
            log_pi_rej = pi_scorer(obs_rej, act_rej)
            log_ref_pref = ref_scorer(obs_pref, act_pref)
            log_ref_rej = ref_scorer(obs_rej, act_rej)

            loss = loss_fn(log_pi_pref, log_pi_rej, log_ref_pref, log_ref_rej)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        train_policy.policy.eval()
        epoch_val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_loader_tqdm:
                
                #TODO : same for here we need to adjust as in train
                
                obs_pref = batch["obs_pref"].to(device)
                act_pref = batch["act_pref"].to(device)
                obs_rej = batch["obs_rej"].to(device)
                act_rej = batch["act_rej"].to(device)

                log_pi_pref = pi_scorer(obs_pref, act_pref)
                log_pi_rej = pi_scorer(obs_rej, act_rej)
                log_ref_pref = ref_scorer(obs_pref, act_pref)
                log_ref_rej = ref_scorer(obs_rej, act_rej)

                val_loss = loss_fn(log_pi_pref, log_pi_rej, log_ref_pref, log_ref_rej)
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # rollout policy
        if epoch % roll_out == 0:
            reward, _ = rollout_policy(policy=train_policy, env_name=env_id)

        # Print and log losses
        tqdm.write(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "rollout_reward": reward,
        })
        
        # save policy checkpoint
        if epoch % save_model == 0:
            train_policy.save(save_dir / f"dpo_finetuned_epoch_{epoch+1}.zip")

    # Finalize wandb logging
    wandb.finish()
















if __name__ == "__main__":
    env_id = "Pendulum-v1"
    total_steps_list = [400_000, 200_000]
    model_names = [
        f"ppo_{env_id.lower().replace('-', '_')}_{steps // 1000}k"
        for steps in total_steps_list
    ]

    pi1_name = model_names[0]
    pi2_name = model_names[1]
    pi1_steps = total_steps_list[0]
    pi2_steps = total_steps_list[1]
    base_dir = Path(__file__).resolve().parent.parent
    dpo_dataset_path = base_dir / f"data/preference_pairs_{pi1_name}_vs_{pi2_name}.pkl"
    policy_path = base_dir / f"checkpoints/{pi1_name}/{pi1_name}_{pi1_steps}_steps"
    ref_policy_path = base_dir / f"checkpoints/{pi2_name}/{pi2_name}_{pi2_steps}_steps"
    save_dir = base_dir / f"checkpoints/dpo_finetuned_policy_{pi1_name}_vs_{pi2_name}"

    align_with_dpo_pipeline(
        env_id=env_id,
        dpo_dataset_path=dpo_dataset_path,
        policy_path=policy_path,
        ref_policy_path=ref_policy_path,
        save_dir=save_dir,
        epochs=50,
        batch_size=64,
        lr=1e-4,
        beta=1.0,
        seed=66,
    )
