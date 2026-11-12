"""Soft actor-critic with critic-owned encoders and Polyak target updates."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .networks import Actor, TwinCritic, configured_encoder


class SAC:
    def __init__(self, config):
        self.config = config
        pooling = config.get('fleet_pooling', 'mean')
        self.encoder = configured_encoder(config)
        self.actor = Actor(context_dim=self.encoder.output_dim, initial_log_std=config.get('initial_log_std'))
        self.critic = TwinCritic(fleet_pooling=pooling, context_dim=self.encoder.output_dim)
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.critic_parameters = list(self.encoder.parameters()) + list(self.critic.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=config['critic_lr'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.log_alpha = nn.Parameter(torch.tensor(config['initial_alpha']).log())
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['alpha_lr'])
        self.updates = 0

    def update(self, batch):
        obs, action, reward, next_obs, done = batch
        with torch.no_grad():
            # Current actor uses the current encoder. Targets use the target encoder.
            next_action, next_log_prob = self.actor(self.encoder(next_obs))
            q1, q2 = self.target_critic(self.target_encoder(next_obs), next_action)
            target = reward + self.config['gamma'] * (1 - done) * (
                torch.minimum(q1, q2) - self.log_alpha.exp() * next_log_prob)
        q1, q2 = self.critic(self.encoder(obs), action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_parameters, self.config['gradient_clip'])
        self.critic_optimizer.step()
        # Recompute after the critic update, detach before BOTH actor and critic heads.
        encoded = tuple(value.detach() for value in self.encoder(obs))
        self.critic.requires_grad_(False)
        sampled, log_prob = self.actor(encoded)
        qa, qb = self.critic(encoded, sampled)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - torch.minimum(qa, qb)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['gradient_clip'])
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        active_dimensions = 1 + obs['ev_present'].sum(dim=1, keepdim=True)
        target_entropy = -self.config['entropy_per_dimension'] * active_dimensions
        alpha_loss = -(self.log_alpha * (log_prob.detach() + target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for source, target_module in ((self.encoder, self.target_encoder),
                                          (self.critic, self.target_critic)):
                for parameter, target_parameter in zip(source.parameters(), target_module.parameters()):
                    target_parameter.lerp_(parameter, self.config['tau'])
        self.updates += 1
        metrics = {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(),
                   'alpha': self.log_alpha.exp().item(), 'q_mean': q1.mean().item()}
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError(f'nonfinite SAC update {metrics}')
        return metrics
