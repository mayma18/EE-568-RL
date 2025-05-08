from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_curve_simple(log_path: Path, output_path: Path):
    """Same as before — omitted for brevity"""
    try:
        data = pd.read_csv(log_path, skiprows=1)
    except Exception as e:
        print(f"[Error] Failed to read log file: {e}")
        return

    data["r"] = pd.to_numeric(data["r"], errors="coerce")
    if data["r"].isnull().all():
        print("[Error] Reward column contains no valid numeric values.")
        return

    timesteps = data["l"].cumsum()
    rewards = data["r"]

    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title(f"PPO Training: {log_path.parent.name}")
    plt.grid()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Saved] Plot saved to: {output_path}")
    plt.close()

def batch_plot_all_logs(logs_root: Path):
    """
    Traverse all subdirectories under logs_root and generate plots
    for any *.monitor.csv files found.
    """
    monitor_files = logs_root.rglob("*.monitor.csv")  # Recursively search

    for log_file in monitor_files:
        save_path = log_file.parent / "reward_curve.png"
        print(f"[Info] Processing: {log_file}")
        plot_reward_curve_simple(log_file, save_path)

if __name__ == "__main__":
    logs_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    batch_plot_all_logs(logs_dir)
