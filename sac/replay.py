"""Fixed-size CPU replay with an independent sampling stream."""
import numpy as np
import torch


class Replay:
    def __init__(self, capacity, n_slots, seed, version=1):
        self.rng = np.random.default_rng(seed)
        shapes = {'timeseries': (3, 20), 'global': (18 if version == 2 else 4,), 'ev': (n_slots, 7 if version == 2 else 4),
                  'ev_present': (n_slots,)}
        self.obs = {key: np.empty((capacity, *shape), np.float32) for key, shape in shapes.items()}
        self.next_obs = {key: np.empty_like(value) for key, value in self.obs.items()}
        self.action = np.empty((capacity, n_slots + 1), np.float32)
        self.reward = np.empty((capacity, 1), np.float32)
        self.done = np.empty((capacity, 1), np.float32)
        self.capacity, self.size, self.position = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, done):
        index = self.position
        for key in self.obs:
            self.obs[key][index] = obs[key]
            self.next_obs[key][index] = next_obs[key]
        self.action[index], self.reward[index], self.done[index] = action, reward, done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        index = self.rng.integers(self.size, size=batch_size)
        obs = {key: torch.from_numpy(value[index]) for key, value in self.obs.items()}
        next_obs = {key: torch.from_numpy(value[index]) for key, value in self.next_obs.items()}
        return (obs, torch.from_numpy(self.action[index]), torch.from_numpy(self.reward[index]),
                next_obs, torch.from_numpy(self.done[index]))
