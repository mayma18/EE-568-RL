import os
import gym
import numpy as np
from evaluator import ModelEvaluator
from plotter import DataPlotter

def main():
    # Initialize the CartPole environment
    env = gym.make("CartPole-v1")

    # Specify the path to the trained models
    model_path = "cartpole-model-evaluator\\models"
    model_files = [f for f in os.listdir(model_path) if f.endswith('.zip')]
    model_paths = [os.path.join(model_path, f) for f in model_files]

    # Initialize the evaluator
    evaluator = ModelEvaluator(model_paths, env)
    models = evaluator.load_models()

    results = {}
    for model_file in model_files:
        model_name = model_file.split('.')[0]
        print(f"Evaluating model: {model_name}")
        model = models[model_file]
        stats = evaluator.run_simulations(model, num_episodes=50)
        results[model_name] = stats

    # Prepare summary_stats for plotting
    summary_stats = {}
    for model_name, stats in results.items():
        summary_stats[model_name] = {
            'avg_reward': stats['average_reward'],
            'var_reward': stats['variance_reward'],
            'avg_displacement': stats['average_displacement'],
            'var_displacement': stats['variance_displacement'],
            'avg_angle': stats['average_angle'],
            'var_angle': stats['variance_angle']
        }

    # Plot the results and save the figure
    plotter = DataPlotter(summary_stats)
    fig = plotter.plot_comparison(show=False)  # 修改plotter支持返回fig对象
    save_path = os.path.join(model_path, "model_comparison.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    main()