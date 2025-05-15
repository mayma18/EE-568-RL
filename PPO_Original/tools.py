import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboard.backend.event_processing import event_accumulator
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from gymnasium.wrappers import RecordVideo

from datetime import datetime
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env

import glob, os, re, pandas as pd
import random
import json

def plot_tensor_result(log_dir):
    # 读取所有标量
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={ 'scalars': 10000 }
    )
    ea.Reload()

    # 拿到 “rollout/ep_rew_mean” 曲线
    rew = ea.Scalars('rollout/ep_rew_mean')
    steps_r = [x.step for x in rew]
    values_r = [x.value for x in rew]

    # 拿到 “rollout/ep_len_mean” 曲线
    length = ea.Scalars('rollout/ep_len_mean')
    steps_l = [x.step for x in length]
    values_l = [x.value for x in length]

    # 创建两个并排子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 子图1：reward
    ax1.plot(steps_r, values_r)
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_xlim(0, np.max(steps_r) + 1000)
    ax1.set_ylim(0, np.max(values_r) + 10)
    ax1.set_title("PPO: Mean Episode Reward")
    ax1.grid(True)

    # 子图2：episode length
    ax2.plot(steps_l, values_l)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Mean Episode Length")
    ax2.set_xlim(0, np.max(steps_l) + 1000)
    ax2.set_ylim(0, np.max(values_l) + 10)
    ax2.set_title("PPO: Mean Episode Length")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()




def test_model(model_type, model_path, n_episodes=5, render=False, record=False, traj=False, file_prefix = None):

    print(model_path)

    # 录像 & csv 路径
    path_list = model_path.split("\\")
    base_dir = "\\".join(path_list[:-1])
    video_path = os.path.join(base_dir, "videos")
    # video_path = os.path.join(base_dir)
    os.makedirs(video_path, exist_ok=True)

    # 根据模型名设定前缀
    if path_list[-1] == "model_full_training.zip" or path_list[-1] == "model_full_training":
        name_prefix_ = "Pi-1"  
    else: 
        name_prefix_ = "Pi-2"

    # 初始化环境的渲染模式
    render_mode = "human" if render else None
    
    if record:
        render_mode = "rgb_array"

    env = gym.make("CartPole-v1", render_mode=render_mode)
    if record:
        env = RecordVideo(
            env,
            video_folder=video_path,
            episode_trigger=lambda ep_id: ep_id < n_episodes,
            name_prefix=name_prefix_
        )
    env = DummyVecEnv([lambda: env])

    print("")

    # 如果要记录轨迹，准备容器


    # 加载模型
    if model_type == "PPO":
        model = PPO.load(model_path, device="cpu", env=env)
    elif model_type == "DQN":
        model = DQN.load(model_path, device="cpu", env=env)
    else:
        raise ValueError("Unsupported model type. Use 'PPO' or 'DQN'.")

    # 开始测试
    for ep in range(n_episodes):
        obs = env.reset()           # NOTE: VecEnv reset
        done = False
        score = 0
        step  = 0

        # 每次清空
        if traj:
            traj_data = []

        while not done:
            action, _ = model.predict(obs)
            obs_next, reward, done, _ = env.step(action)
            score += reward[0]
            
            if traj:
                # 展平 obs（假设 shape=(1,4)），取第 0 个环境
                o = obs[0]
                traj_data.append({
                    "episode": ep + 1,
                    "step": step,
                    "x": float(o[0]),
                    "x_dot": float(o[1]),
                    "theta": float(o[2]),
                    "theta_dot": float(o[3]),
                    "action": int(action[0]),
                    "reward": float(reward[0])
                })

            obs = obs_next
            step += 1

        # 输出轨迹到 CSV
        if traj:
            df = pd.DataFrame(traj_data)
            csv_name = f"{name_prefix_}-trajectory-{ep}.csv"
            csv_path = os.path.join(base_dir, "trajectory", csv_name)

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            df.to_csv(csv_path, index=False)
            print(f"Trajectory saved to: {csv_path}")


        print(f"Episode: {ep+1} Score: {score}")

    env.close()




# 原始颜色列表，用于不同目录的区分
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue',
          'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon',
          'gold', 'lightpurple', 'darkred', 'darkblue']


def lighten_color(color, amount=0.5):
    """
    将给定颜色与白色混合以获得更浅的色调。
    :param color: matplotlib 支持的颜色字符串
    :param amount: 混合比例，0-1，值越大越接近白色
    """
    rgb = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    return tuple(rgb + (white - rgb) * amount)


def load_monitor_results(path: str) -> pd.DataFrame:
    """
    读取指定目录下所有 monitor.csv 日志并合并成一个 DataFrame，按时间排序并重置 t 使最早为 0。
    """
    pattern = os.path.join(path, "*.monitor.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No monitor files found in {path}")

    dfs = []
    t_starts = []
    for fn in files:
        with open(fn, 'r') as f:
            header = json.loads(f.readline().lstrip('#'))
            t_starts.append(header.get('t_start', 0))
            df = pd.read_csv(f)
            df['t'] += header.get('t_start', 0)
            dfs.append(df)
    result = pd.concat(dfs, ignore_index=True).sort_values('t').reset_index(drop=True)
    result['t'] -= min(t_starts)
    return result


def plot_result(dirs, num_timesteps=None, xaxis='timesteps', task_name='', window=100):
    """
    简化版绘图：上下两个子图分别显示 Episode Rewards 和 Episode Length，保留原始颜色，并使点和线有细微色差。
    :param dirs: 日志目录列表
    :param num_timesteps: 最大 timesteps（按 episode 长度累计）
    :param xaxis: 'timesteps' | 'episodes' | 'walltime_hrs'
    :param task_name: 图标题前缀
    :param window: 平滑窗口大小
    """
    series = []
    for idx, folder in enumerate(dirs):
        df = load_monitor_results(folder)
        if num_timesteps is not None:
            df = df[df['l'].cumsum() <= num_timesteps]
        if xaxis == 'timesteps':
            df['x'] = df['l'].cumsum()
        elif xaxis == 'episodes':
            df['x'] = np.arange(len(df))
        elif xaxis == 'walltime_hrs':
            df['x'] = df['t'] / 3600.0
        else:
            raise ValueError(f"Unknown xaxis: {xaxis}")
        series.append((df, os.path.basename(folder), COLORS[idx % len(COLORS)]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    for df, label, color in series:
        light_color = lighten_color(color, 0.5)
        
        # Episode Rewards                 s=2
        axes[0].scatter(df['x'], df['r'], s=2, label=f'{label} raw')   #  c=light_color, alpha=0.8,
        if len(df) >= window:
            smoothed = df['r'].rolling(window).mean().iloc[window-1:] 
            axes[0].plot(df['x'].iloc[window-1:], smoothed, c=color, linewidth=1.5, label=f'{label} {window}-step avg')
        
        # Episode Length                  s=2
        axes[1].scatter(df['x'], df['l'], s=2, label=f'{label} raw')   #  c=light_color, alpha=0.8,
        if len(df) >= window:
            smoothed_l = df['l'].rolling(window).mean().iloc[window-1:]
            axes[1].plot(df['x'].iloc[window-1:], smoothed_l, c=color, linewidth=1.5, label=f'{label} {window}-step avg')

    axes[0].set_title(f'{task_name} Rewards')
    axes[0].set_ylabel('Episode Rewards')
    axes[0].grid(True)
    axes[1].set_title(f'{task_name} Episode Length')
    axes[1].set_xlabel(xaxis)
    axes[1].set_ylabel('Episode Length')
    axes[1].grid(True)

    # 图例
    for ax in axes:
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.show()




class RLHF_class:
    def __init__(self, path):

        # 初始化数据
        self.path = path
        self.reward_pi_1 = []
        self.reward_pi_2 = []
        self.traj_list_1 = []
        self.traj_list_2 = []

        self.traj_prefer = []
        self.traj_reject = []
        self.choose_probs = None

        # 启动函数
        self.read_pattern()             # 一次性扫 Pi-1, Pi-2, … 各类文件
        self.compute_total_reward()     # 计算每个 Pi 的总奖励
        self.comput_RLHF_prob(self.reward_pi_1, self.reward_pi_2)
        self.store_trajectory_and_action_pair()         # 将轨迹数据存储到 traj_list_1, traj_list_2 中
        self.select_prefer_and_reject_traj()
        self.make_csv()                # 将数据存储到 csv 文件中

    def read_pattern(self):

        # 1. 扫描所有 Pi-*-trajectory-*.csv
        pattern = os.path.join(self.path, 'Pi-*-trajectory-*.csv')
        all_files = glob.glob(pattern)

        # 2. 按前缀 Pi-1, Pi-2… 分组
        groups = {}
        for fp in all_files:
            name = os.path.basename(fp)
            m = re.match(r'(Pi-\d+)-trajectory-.*\.csv', name)
            if not m:
                continue
            prefix = m.group(1)               # 比如 "Pi-1"
            groups.setdefault(prefix, []).append(fp)

        # 3. 为每个分组创建属性 data_base_1, data_base_2, …
        for idx, prefix in enumerate(sorted(groups.keys()), start=1):
            dfs = [pd.read_csv(f) for f in groups[prefix]]
            setattr(self, f"data_base_{idx}", dfs)
    
    def compute_total_reward(self):

        for data in self.data_base_1:
            reward = np.sum(data['reward'])
            self.reward_pi_1.append(reward)

        for data in self.data_base_2:
            reward = np.sum(data['reward'])
            self.reward_pi_2.append(reward)
    
    def store_trajectory_and_action_pair(self):

        self.traj_list_1 = []
        self.traj_list_2 = []

        for data in self.data_base_1:
            
            x         = np.array(data['x'])
            x_dot     = np.array(data['x_dot'])
            theta     = np.array(data['theta'])
            theta_dot = np.array(data['theta_dot'])


            state     = np.array([x, x_dot, theta, theta_dot])
            action    = np.array(data['action'])

            traj = []
            for i in range(state.shape[1]):
                traj.append({"state": state[:, i], "action": action[i]})
            
            self.traj_list_1.append(traj)
        
        
        for data in self.data_base_2:
                
            x         = np.array(data['x'])
            x_dot     = np.array(data['x_dot'])
            theta     = np.array(data['theta'])
            theta_dot = np.array(data['theta_dot'])

            state     = np.array([x, x_dot, theta, theta_dot])
            action    = np.array(data['action'])

            traj = []
            for i in range(state.shape[1]):
                traj.append({"state": state[:, i], "action": action[i]})
            
            self.traj_list_2.append(traj)
    
    def comput_RLHF_prob(self, reward1, reward2):
        """
        计算 RLHF 概率
        Compute RLHF probability
        """
        self.choose_probs = np.exp(reward1) / (np.exp(reward1) + np.exp(reward2))

    def select_prefer_and_reject_traj(self)->None:
        
        for index, prob in enumerate(self.choose_probs):
            sign = random.choices([0, 1], weights=[prob, 1 - prob])[0] # 选择 0 或 1 的概率
            if sign == 0:
                self.traj_prefer.append(self.traj_list_1[index])
                self.traj_reject.append(self.traj_list_2[index])
            else:
                self.traj_prefer.append(self.traj_list_2[index])
                self.traj_reject.append(self.traj_list_1[index])

    def make_csv(self) -> None:
        data = []
        for i in range(len(self.traj_prefer)):
            # 清洗 preferred 轨迹
            clean_pref = []
            for step in self.traj_prefer[i]:
                clean_pref.append({
                    'state': step['state'].tolist(),      # array -> list
                    'action': int(step['action'])         # np.int64 -> int
                })
            # 清洗 rejected 轨迹
            clean_rej = []
            for step in self.traj_reject[i]:
                clean_rej.append({
                    'state': step['state'].tolist(),
                    'action': int(step['action'])
                })

            data.append({
                "episode": i,
                # 把 list[dict] 序列化为标准 JSON
                "preferred": json.dumps(clean_pref, ensure_ascii=False),
                "rejected": json.dumps(clean_rej, ensure_ascii=False),
            })

        df = pd.DataFrame(data)
        parent_dir = os.path.abspath(os.path.join(self.path, os.pardir))
        df.to_csv(os.path.join(parent_dir, "RLHF_trajectory_pairs.csv"), index=False, encoding='utf-8')
        print("RLHF_trajectory_pairs.csv 已保存到：", parent_dir)