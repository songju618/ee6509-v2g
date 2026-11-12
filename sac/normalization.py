"""Frozen training-only feature statistics, serialized in every checkpoint."""
from __future__ import annotations

import numpy as np
import torch

from .physical import engineered_observation


class Normalizer:
    def __init__(self, statistics):
        self.statistics = statistics

    @classmethod
    def fit(cls, observations, version=1):
        values = {key: [] for key in ('timeseries', 'global', 'ev')}
        for obs in observations:
            if version == 2:
                obs = engineered_observation(obs)
            values['timeseries'].append(obs['timeseries'].T)
            values['global'].append(obs['global'][None])
            values['ev'].append(obs['ev'][obs['ev_present'].astype(bool)])
        statistics = {}
        for key, rows in values.items():
            array = np.concatenate(rows).astype(np.float64)
            if version == 2:
                center = np.median(array, axis=0)
                scale = np.maximum(np.quantile(array, .75, axis=0) - np.quantile(array, .25, axis=0), .05)
            else:
                center, scale = array.mean(axis=0), np.maximum(array.std(axis=0), .01)
            statistics[key] = {'mean': center.tolist(), 'std': scale.tolist()}
        if version == 2:
            statistics['version'] = 2
        return cls(statistics)

    def transform(self, obs):
        if self.statistics.get('version', 1) == 2:
            obs = engineered_observation(obs)
        result = {}
        for key in ('timeseries', 'global', 'ev'):
            stats = self.statistics[key]
            mean = np.asarray(stats['mean'], dtype=np.float32)
            std = np.asarray(stats['std'], dtype=np.float32)
            if key == 'timeseries':
                mean, std = mean[:, None], std[:, None]
            result[key] = (np.asarray(obs[key], dtype=np.float32) - mean) / std
        result['ev_present'] = np.asarray(obs['ev_present'], dtype=np.float32)
        result['ev'] *= result['ev_present'][..., None]
        return result

    def tensor(self, obs):
        return {key: torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)
                for key, value in self.transform(obs).items()}
