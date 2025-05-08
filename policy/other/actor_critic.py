import torch

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net
        self.value_net = sb3_policy.value_net
        self.action_dist = sb3_policy.action_dist
        self.log_std = sb3_policy.log_std
        self.action_space = sb3_policy.action_space

    def forward(self, obs):
        # Shared encoder
        x = self.features_extractor(obs)

        # Actor & Critic latent
        latent_pi, latent_vf = self.mlp_extractor(x)

        # Actor: mean & dist
        mean_actions = self.action_net(latent_pi)
        dist = self.action_dist.proba_distribution(mean_actions, self.log_std)

        # Critic: state value
        value = self.value_net(latent_vf).squeeze(-1)  # shape: [batch] or scalar

        return dist, value

    def predict(self, obs, deterministic=True):
        dist, _ = self.forward(obs)
        action = dist.mode() if deterministic else dist.sample()
        low = torch.tensor(self.action_space.low, dtype=action.dtype, device=action.device)
        high = torch.tensor(self.action_space.high, dtype=action.dtype, device=action.device)
        return torch.clamp(action, low, high)