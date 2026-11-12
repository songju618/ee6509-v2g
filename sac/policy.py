"""Standalone deterministic/stochastic inference entry point for C's runner."""
from __future__ import annotations

from pathlib import Path
import hashlib
from numbers import Integral

import numpy as np
import torch

from .networks import Actor, configured_encoder
from .normalization import Normalizer
from .physical import environment_action


class Policy:
    def __init__(self, checkpoint, n_slots):
        if isinstance(n_slots, (bool, np.bool_)) or not isinstance(n_slots, Integral) or not 1 <= n_slots <= 10:
            raise ValueError('n_slots must be an integer from 1 to 10')
        self.n_slots = int(n_slots)
        config = checkpoint.get('metadata', {}).get('config', {})
        self.anchored = config.get('action_mapping') == 'zero_power'
        self.encoder = configured_encoder(config).eval()
        self.actor = Actor(context_dim=self.encoder.output_dim).eval()
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.normalizer = Normalizer(checkpoint['normalizer'])
        self.metadata = checkpoint.get('metadata', {})

    def _observation(self, obs):
        shapes = {'timeseries': (3, 20), 'global': (4,),
                  'ev': (self.n_slots, 4), 'ev_present': (self.n_slots,)}
        for key, shape in shapes.items():
            array = np.asarray(obs[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f'{key} must be finite with shape {shape}')
        return self.normalizer.tensor(obs)

    @torch.inference_mode()
    def act(self, obs, deterministic=True):
        encoded = self.encoder(self._observation(obs))
        action, _ = self.actor(encoded, deterministic=deterministic)
        internal = action[0].numpy().copy()
        return environment_action(internal, obs) if self.anchored else internal

    @torch.inference_mode()
    def attention(self, obs):
        return self.encoder(self._observation(obs))[-1][0].numpy().copy()


def load_policy(path, n_slots):
    """Load a trusted local checkpoint on CPU without constructing an environment."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') not in (1, 2):
        raise ValueError('unsupported checkpoint format')
    policy = Policy(checkpoint, n_slots)
    policy.checkpoint_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return policy
