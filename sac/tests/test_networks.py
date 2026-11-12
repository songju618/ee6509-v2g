import unittest

import numpy as np
import torch

from sac.agent import SAC
from sac.networks import Actor, Encoder, TwinCritic
from sac.normalization import Normalizer
from sac.protocol import configure


class NetworkTests(unittest.TestCase):
    def setUp(self):
        configure()
        self.encoder = Encoder(fleet_pooling='scaled_sum')
        self.actor = Actor()
        self.critic = TwinCritic(fleet_pooling='scaled_sum')

    def observation(self, n=5, empty=False):
        present = torch.ones(4, n)
        present[1, 1::2] = 0
        present[2] = 0
        if empty:
            present.zero_()
        ev = torch.randn(4, n, 4) * present.unsqueeze(-1)
        return {'timeseries': torch.randn(4, 3, 20), 'global': torch.randn(4, 4),
                'ev': ev, 'ev_present': present}

    def test_shapes_bounds_masks_and_shared_n5_n10_parameters(self):
        parameters = sum(p.numel() for module in (self.encoder, self.actor, self.critic) for p in module.parameters())
        for n in (5, 10):
            obs = self.observation(n)
            encoded = self.encoder(obs)
            action, log_prob = self.actor(encoded)
            self.assertEqual(action.shape, (4, n + 1))
            self.assertEqual(log_prob.shape, (4, 1))
            self.assertTrue((action.abs() <= 1).all())
            self.assertTrue((action[:, 1:][~obs['ev_present'].bool()] == 0).all())
            self.assertTrue((encoded[1][~obs['ev_present'].bool()] == 0).all())
            for q in self.critic(encoded, action):
                self.assertEqual(q.shape, (4, 1))
            self.assertEqual(parameters, sum(p.numel() for module in (self.encoder, self.actor, self.critic)
                                             for p in module.parameters()))

    def test_permutation_equivariance_and_critic_invariance(self):
        obs = self.observation(10)
        permutation = torch.randperm(10)
        permuted = dict(obs, ev=obs['ev'][:, permutation], ev_present=obs['ev_present'][:, permutation])
        encoded = self.encoder(obs)
        new_encoded = self.encoder(permuted)
        action, log_prob = self.actor(encoded, deterministic=True)
        new_action, new_log_prob = self.actor(new_encoded, deterministic=True)
        torch.testing.assert_close(new_action[:, :1], action[:, :1])
        torch.testing.assert_close(new_action[:, 1:], action[:, 1:][:, permutation])
        torch.testing.assert_close(new_log_prob, log_prob)
        torch.testing.assert_close(new_encoded[-1], encoded[-1][:, permutation][:, :, permutation])
        for first, second in zip(self.critic(encoded, action), self.critic(new_encoded, new_action)):
            torch.testing.assert_close(first, second)

    def test_empty_forward_and_backward_are_finite(self):
        for n in (5, 10):
            obs = self.observation(n, empty=True)
            encoded = self.encoder(obs)
            action, log_prob = self.actor(encoded)
            q1, q2 = self.critic(encoded, action)
            loss = (q1 + q2 + log_prob).sum()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue((encoded[-1] == 0).all())
            self.assertTrue((action[:, 1:] == 0).all())
            for module in (self.encoder, self.actor, self.critic):
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_masked_slots_cannot_change_active_actions_or_q(self):
        obs = self.observation()
        altered = dict(obs, ev=obs['ev'].clone())
        altered['ev'][~obs['ev_present'].bool()] = 1e3
        encoded, altered_encoded = self.encoder(obs), self.encoder(altered)
        action = self.actor(encoded, deterministic=True)[0]
        altered_action = self.actor(altered_encoded, deterministic=True)[0]
        torch.testing.assert_close(action, altered_action)
        for first, second in zip(self.critic(encoded, action), self.critic(altered_encoded, altered_action)):
            torch.testing.assert_close(first, second)

    def test_actor_loss_does_not_update_encoder(self):
        obs = self.observation()
        before = [p.detach().clone() for p in self.encoder.parameters()]
        encoded = tuple(t.detach() for t in self.encoder(obs))
        action, log_prob = self.actor(self.encoder(obs))
        q1, q2 = self.critic(encoded, action)
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=.01)
        optimizer.zero_grad()
        (.1 * log_prob - torch.minimum(q1, q2)).mean().backward()
        optimizer.step()
        for old, parameter in zip(before, self.encoder.parameters()):
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(old, parameter)
        self.assertTrue(any(p.grad is not None for p in self.actor.parameters()))

    def test_complete_sac_update_with_empty_fleet_updates_critic_encoder(self):
        import json
        from pathlib import Path
        config = dict(json.loads(Path('sac/configs/default.json').read_text()), observation_version=1)
        agent = SAC(config)
        before = [p.detach().clone() for p in agent.encoder.parameters()]
        obs = self.observation(empty=True)
        metrics = agent.update((obs, torch.zeros(4, 6), torch.randn(4, 1), obs, torch.zeros(4, 1)))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(old, parameter) for old, parameter in zip(before, agent.encoder.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.encoder.parameters()))

    def test_fleet_pooling_preserves_occupied_count(self):
        first = self.observation(5)
        first['ev_present'].zero_()
        first['ev_present'][:, :2] = 1
        first['ev'][:] = 0
        first['ev'][:, :2] = torch.tensor([1., .5, 12., 4.])
        second = dict(first)
        second['ev_present'] = torch.zeros(4, 10)
        second['ev_present'][:, :4] = 1
        second['ev'] = torch.zeros(4, 10, 4)
        second['ev'][:, :4] = torch.tensor([1., .5, 12., 4.])
        original = self.encoder(first)[0][:, -64:]
        doubled = self.encoder(second)[0][:, -64:]
        torch.testing.assert_close(doubled, 2 * original)
        self.assertGreater(original.abs().sum().item(), 0.)

    def test_normalization_keeps_empty_slots_zero(self):
        stats = {key: {'mean': [1.] * size, 'std': [2.] * size}
                 for key, size in [('timeseries', 3), ('global', 4), ('ev', 4)]}
        obs = {'timeseries': np.zeros((3, 20)), 'global': np.zeros(4),
               'ev': np.zeros((5, 4)), 'ev_present': np.zeros(5)}
        normalized = Normalizer(stats).transform(obs)
        self.assertTrue((normalized['ev'] == 0).all())
        self.assertTrue(np.isfinite(normalized['timeseries']).all())


if __name__ == '__main__':
    unittest.main()
