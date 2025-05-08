import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryScorer:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, obs, act):
        """
        obs: [B, T, obs_dim]
        act: [B, T, act_dim]
        Returns: log prob sum for each trajectory, shape: [B]
        """
        B, T, _ = obs.shape
        obs_flat = obs.view(B * T, -1)
        act_flat = act.view(B * T, -1)
        dist = self.policy.get_distribution(obs_flat)
        log_probs = dist.log_prob(act_flat)
        if log_probs.ndim == 2:  # [B*T, act_dim]
            log_probs = log_probs.sum(dim=-1)
        return log_probs.view(B, T).sum(dim=1)           # [B]

class DPOTrajectoryLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, log_pi_pref, log_pi_rej, log_ref_pref, log_ref_rej):
        """
        All inputs: [B]
        """
        logits = self.beta * ((log_pi_pref - log_ref_pref) - (log_pi_rej - log_ref_rej))
        return -F.logsigmoid(logits).mean()
