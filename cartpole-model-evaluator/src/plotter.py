import matplotlib.pyplot as plt
import numpy as np

class DataPlotter:
    def __init__(self, summary_stats):
        self.summary_stats = summary_stats
        self.model_names = list(summary_stats.keys())

    def plot_comparison(self, show=True):
        avg_rewards = [self.summary_stats[m]['avg_reward'] for m in self.model_names]
        var_rewards = [self.summary_stats[m]['var_reward'] for m in self.model_names]
        avg_displacements = [self.summary_stats[m]['avg_displacement'] for m in self.model_names]
        var_displacements = [self.summary_stats[m]['var_displacement'] for m in self.model_names]
        avg_angles = [self.summary_stats[m]['avg_angle'] for m in self.model_names]
        var_angles = [self.summary_stats[m]['var_angle'] for m in self.model_names]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # 奖励
        ax1 = axs[0]
        ax2 = ax1.twinx()
        ax1.bar(self.model_names, avg_rewards, color='C0', alpha=0.7)
        ax2.plot(self.model_names, var_rewards, color='C1', marker='o', linestyle='--', label='Variance')
        ax1.set_title('Average Total Rewards')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Reward', color='C0')
        ax2.set_ylabel('Variance', color='C1')
        ax1.set_xticklabels(self.model_names, rotation=45)
        ax1.grid(axis='y')
        ax2.tick_params(axis='y', labelcolor='C1')

        # 位移
        ax3 = axs[1]
        ax4 = ax3.twinx()
        ax3.bar(self.model_names, avg_displacements, color='C0', alpha=0.7)
        ax4.plot(self.model_names, var_displacements, color='C1', marker='o', linestyle='--', label='Variance')
        ax3.set_title('Average Displacement')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Average Displacement', color='C0')
        ax4.set_ylabel('Variance', color='C1')
        ax3.set_xticklabels(self.model_names, rotation=45)
        ax3.grid(axis='y')
        ax4.tick_params(axis='y', labelcolor='C1')

        # 角度
        ax5 = axs[2]
        ax6 = ax5.twinx()
        ax5.bar(self.model_names, avg_angles, color='C0', alpha=0.7)
        ax6.plot(self.model_names, var_angles, color='C1', marker='o', linestyle='--', label='Variance')
        ax5.set_title('Average Angle')
        ax5.set_xlabel('Model')
        ax5.set_ylabel('Average Angle', color='C0')
        ax6.set_ylabel('Variance', color='C1')
        ax5.set_xticklabels(self.model_names, rotation=45)
        ax5.grid(axis='y')
        ax6.tick_params(axis='y', labelcolor='C1')

        plt.tight_layout()
        if show:
            plt.show()
        return fig