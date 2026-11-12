"""CNN and masked fleet attention with shared actor and critic slot weights."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, width=64, fleet_pooling='mean', ev_features=4, global_features=4):
        super().__init__()
        if fleet_pooling not in ('mean', 'scaled_sum'):
            raise ValueError('fleet_pooling must be mean or scaled_sum')
        self.fleet_pooling = fleet_pooling
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1), nn.SiLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(32 * 20, width), nn.LayerNorm(width), nn.SiLU(),
        )
        self.ev_embed = nn.Linear(ev_features, width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                         nn.Linear(width * 2, width))
        self.final_norm = nn.LayerNorm(width)
        self.output_dim = width * 2 + global_features

    def forward(self, obs):
        present = obs['ev_present'].bool()
        embedded = self.ev_embed(obs['ev'])
        # A finite dummy key prevents all-masked softmax in BOTH forward/backward.
        # Multiplication after an all-masked attention call cannot remove NaNs.
        safe_present = present.clone()
        safe_present[:, 0] |= ~present.any(dim=1)
        attended, weights = self.attention(
            embedded, embedded, embedded, key_padding_mask=~safe_present,
            need_weights=True, average_attn_weights=True,
        )
        slots = self.norm(embedded + attended)
        slots = self.final_norm(slots + self.feedforward(slots))
        slots = slots * present.unsqueeze(-1)
        divisor = present.sum(dim=1, keepdim=True).clamp_min(1) if self.fleet_pooling == 'mean' else 5.0
        pooled = slots.sum(dim=1) / divisor
        context = torch.cat((self.cnn(obs['timeseries']), obs['global'], pooled), dim=-1)
        weights = weights * present.unsqueeze(1) * present.unsqueeze(2)
        return context, slots, present, weights


class Actor(nn.Module):
    def __init__(self, context_dim=132, width=64, initial_log_std=None):
        super().__init__()
        self.bess = nn.Sequential(nn.Linear(context_dim, 128), nn.SiLU(),
                                  nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
        self.ev = nn.Sequential(nn.Linear(context_dim + width, 128), nn.SiLU(),
                                nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

        if initial_log_std is not None:
            for head in (self.bess, self.ev):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                head[-1].bias.data[1] = initial_log_std

    def forward(self, encoded, deterministic=False):
        context, slots, present, _ = encoded
        # The encoder belongs exclusively to the critic optimizer.
        context, slots = context.detach(), slots.detach()
        shared = context.unsqueeze(1).expand(-1, slots.shape[1], -1)
        outputs = torch.cat((self.bess(context).unsqueeze(1),
                             self.ev(torch.cat((shared, slots), dim=-1))), dim=1)
        mean, log_std = outputs.unbind(dim=-1)
        log_std = log_std.clamp(-5., 1.)
        sample = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        action = sample.tanh()
        mask = torch.cat((torch.ones_like(present[:, :1]), present), dim=1)
        # Stable tanh correction, including saturated means.
        gaussian = -.5 * (((sample - mean) / log_std.exp()).square()
                          + 2 * log_std + math.log(2 * math.pi))
        correction = 2 * (math.log(2.) - sample - F.softplus(-2 * sample))
        log_prob = ((gaussian - correction) * mask).sum(dim=-1, keepdim=True)
        return action * mask, log_prob


class QHead(nn.Module):
    def __init__(self, context_dim=132, width=64, fleet_pooling='mean'):
        super().__init__()
        if fleet_pooling not in ('mean', 'scaled_sum'):
            raise ValueError('fleet_pooling must be mean or scaled_sum')
        self.fleet_pooling = fleet_pooling
        self.slot_action = nn.Sequential(nn.Linear(width + 1, 64), nn.SiLU(),
                                         nn.Linear(64, 64), nn.SiLU())
        self.value = nn.Sequential(nn.Linear(context_dim + 1 + 64, 128), nn.SiLU(),
                                   nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, encoded, action):
        context, slots, present, _ = encoded
        pairs = torch.cat((slots, action[:, 1:].unsqueeze(-1)), dim=-1)
        pooled = (self.slot_action(pairs) * present.unsqueeze(-1)).sum(dim=1)
        divisor = present.sum(dim=1, keepdim=True).clamp_min(1) if self.fleet_pooling == 'mean' else 5.0
        pooled = pooled / divisor
        return self.value(torch.cat((context, action[:, :1], pooled), dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, fleet_pooling='mean', context_dim=132):
        super().__init__()
        self.q1 = QHead(fleet_pooling=fleet_pooling, context_dim=context_dim)
        self.q2 = QHead(fleet_pooling=fleet_pooling, context_dim=context_dim)

    def forward(self, encoded, action):
        return self.q1(encoded, action), self.q2(encoded, action)


def configured_encoder(config):
    version = config.get('observation_version', 1)
    return Encoder(fleet_pooling=config.get('fleet_pooling', 'mean'),
                   ev_features=7 if version == 2 else 4,
                   global_features=18 if version == 2 else 4)
