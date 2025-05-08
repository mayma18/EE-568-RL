from train_sb3_policy import train_sb3_policy

if __name__ == "__main__":
    # Train reference (stronger) policy
    ref_path = train_sb3_policy(
        env_id="Pendulum-v1",
        algo="ppo",
        total_timesteps=400_000,
        save_path="checkpoints/ppo_ref"
    )

    # Train trainable (weaker) policy
    train_path = train_sb3_policy(
        env_id="Pendulum-v1",
        algo="ppo",
        total_timesteps=200_000,
        save_path="checkpoints/ppo_train"
    )

    from align_with_dpo_pipeline import align_with_dpo_pipeline

    align_with_dpo_pipeline(
        env_id="Pendulum-v1",
        dpo_dataset_path="data/trajectory_pairs.csv",
        policy_path=train_path,
        ref_policy_path=ref_path,
        save_dir="checkpoints/dpo_finetuned",
        policy_algo="ppo",
        ref_policy_algo="ppo",
        epochs=30,
        batch_size=64,
        lr=1e-4,
        beta=1.0,
        seed=123,
    )
