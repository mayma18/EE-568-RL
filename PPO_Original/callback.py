import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnRewardThreshold(BaseCallback):
    """
    when the reward of a single episode reaches a certain threshold, save the model.
    当单幕回报 ≥ threshold 时，只保存一次模型
    """
    def __init__(self, threshold: float, save_path: str, verbose=1):
        super().__init__(verbose)
        self.threshold = threshold
        self.save_path = save_path
        self._already_saved = False

    def _on_step(self) -> bool:
        # 通过 Monitor wrapper，每个 env.step() 后 info 字典里可能包含 'episode'
        infos = self.locals.get('infos', [])
        for info in infos:
            # Monitor 在 episode 结束时会把 {'episode': {'r': reward, 'l': length, ...}} 放到 info 里
            ep = info.get('episode')
            if ep is not None:
                ep_reward = ep.get('r', 0.0)
                if ep_reward >= self.threshold and not self._already_saved:
                    # 保存模型
                    os.makedirs(self.save_path, exist_ok=True)
                    path = os.path.join(self.save_path, f"model_at_reward_{int(self.threshold)}.zip")
                    self.model.save(path)
                    if self.verbose > 0:
                        print(f"[SaveOnRewardThreshold] reward={ep_reward:.2f} ≥ {self.threshold}, saved to {path}")
                    self._already_saved = True
                    # 如果只想保存一次，就可以在这儿返回 False 来停止后续回调
                    # return False
        return True
