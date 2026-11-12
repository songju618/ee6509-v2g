import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from env import V2GEnv
from sac.agent import SAC
from sac.normalization import Normalizer
from sac.physical import environment_action
from sac.policy import load_policy
from sac.protocol import configure
from sac.replay import Replay
from sac.train import training_observations


class PolicyArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        cls.config = json.loads(Path('sac/configs/default.json').read_text())
        cls.normalizer = Normalizer.fit(training_observations(10, [0], True), version=2)

    def test_all_physical_slot_counts_and_empty_fleets(self):
        agent = SAC(self.config)
        state = {'format_version': 2, 'encoder': agent.encoder.state_dict(), 'actor': agent.actor.state_dict(),
                 'normalizer': self.normalizer.statistics, 'metadata': {'config': self.config}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.pt'
            torch.save(state, path)
            for n in range(1, 11):
                policy = load_policy(path, n)
                environment = V2GEnv(n_slots=n)
                for date in ('2026-10-01', '2026-10-03'):
                    obs, _ = environment.reset(seed=0, options={'date': date})
                    for _ in range(48):
                        action = policy.act(obs)
                        self.assertEqual(action.shape, (n + 1,))
                        self.assertEqual(action.dtype, np.float32)
                        self.assertTrue(np.isfinite(action).all())
                        self.assertTrue((action[1:][obs['ev_present'] == 0] == 0).all())
                        obs, _, _, _, _ = environment.step(action)

    def test_internal_action_replay_and_empty_fleet_backward(self):
        agent = SAC(self.config)
        environment = V2GEnv(n_slots=10)
        obs, _ = environment.reset(seed=0, options={'date': '2026-10-03'})
        for _ in range(2):
            obs, _, _, _, _ = environment.step(np.r_[-1., np.zeros(10)])
        normalized = self.normalizer.transform(obs)
        replay = Replay(10, 10, 0, version=2)
        internal = np.zeros(11, np.float32)
        external = environment_action(internal, obs)
        self.assertAlmostEqual(float(external[0]), -1., places=6)
        self.assertNotEqual(float(external[0]), float(internal[0]))
        replay.add(normalized, internal, 0., normalized, False)
        batch = replay.sample(4)
        np.testing.assert_array_equal(batch[1].numpy(), np.zeros((4, 11)))
        mean_action, log_probability = agent.actor(agent.encoder(batch[0]), deterministic=True)
        torch.testing.assert_close(mean_action, torch.zeros_like(mean_action))
        expected = 2. - .5 * np.log(2 * np.pi)
        torch.testing.assert_close(log_probability, torch.full_like(log_probability, expected))
        metrics = agent.update(batch)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        encoded = agent.encoder(batch[0])
        before = [p.detach().clone() for p in agent.encoder.parameters()]
        for parameter in agent.encoder.parameters():
            parameter.grad = None
        action, log_prob = agent.actor(encoded)
        qa, qb = agent.critic(tuple(t.detach() for t in encoded), action)
        optimizer = torch.optim.Adam(agent.actor.parameters(), lr=.001)
        optimizer.zero_grad()
        (log_prob - qa - qb).mean().backward()
        optimizer.step()
        for old, parameter in zip(before, agent.encoder.parameters()):
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(old, parameter)

    def test_empty_padding_preserves_active_predictions(self):
        agent = SAC(self.config)
        for head in (agent.actor.bess, agent.actor.ev):
            torch.nn.init.normal_(head[-1].weight, std=.05)
        environment = V2GEnv(n_slots=1)
        obs, _ = environment.reset(seed=0, options={'date': '2026-10-01'})
        for _ in range(18):
            obs, _, _, _, _ = environment.step(np.zeros(2))
        padded = dict(obs, ev=np.vstack([obs['ev'], np.zeros((9, 4), np.float32)]),
                      ev_present=np.r_[obs['ev_present'], np.zeros(9, np.int8)])
        first = agent.actor(agent.encoder(self.normalizer.tensor(obs)), deterministic=True)[0]
        second = agent.actor(agent.encoder(self.normalizer.tensor(padded)), deterministic=True)[0]
        torch.testing.assert_close(first, second[:, :2])
        self.assertTrue((second[:, 2:] == 0).all())

    def test_engineered_policy_is_slot_permutation_equivariant(self):
        agent = SAC(self.config)
        for head in (agent.actor.bess, agent.actor.ev):
            torch.nn.init.normal_(head[-1].weight, std=.05)
        environment = V2GEnv(n_slots=5)
        obs, _ = environment.reset(seed=0, options={'date': '2026-10-01'})
        for _ in range(20):
            obs, _, _, _, _ = environment.step(np.zeros(6))
        permutation = np.array([3, 0, 4, 1, 2])
        changed = dict(obs, ev=obs['ev'][permutation], ev_present=obs['ev_present'][permutation])
        first = agent.actor(agent.encoder(self.normalizer.tensor(obs)), deterministic=True)[0]
        second = agent.actor(agent.encoder(self.normalizer.tensor(changed)), deterministic=True)[0]
        torch.testing.assert_close(first[:, :1], second[:, :1])
        torch.testing.assert_close(first[:, 1:][:, permutation], second[:, 1:])


if __name__ == '__main__':
    unittest.main()
