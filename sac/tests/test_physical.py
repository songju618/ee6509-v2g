import unittest

import numpy as np

from env import V2GEnv
from sac.physical import anchored_power, engineered_observation, environment_action, feasible_from_observation
from sac.scenarios import extended_scenario


class PhysicalFeatureTests(unittest.TestCase):
    def test_closed_intervals_match_environment_for_complete_trajectories(self):
        maximum = 0.
        rng = np.random.default_rng(6509)
        for n in (1, 5, 10):
            environment = V2GEnv(n_slots=n)
            for date in ('2026-10-01', '2026-10-03'):
                for seed in (0, 20, 80):
                    scenario = extended_scenario(date, seed, n_slots=n)
                    obs, _ = environment.reset(options={'scenario': scenario})
                    for t in range(48):
                        expected = environment.feasible_intervals()
                        reconstructed = feasible_from_observation(obs)
                        error = np.abs(expected - reconstructed).max()
                        maximum = max(maximum, error)
                        self.assertLess(error, 1e-5, (n, date, seed, t))
                        u = rng.uniform(-1, 1, n + 1)
                        action = environment_action(u, obs)
                        obs, _, _, _, _ = environment.step(action)
        self.assertLess(maximum, 1e-5)

    def test_zero_anchor_endpoints_and_fixed_intervals(self):
        intervals = np.array([[-60., 60.], [0., 60.], [-60., 0.], [2., 6.], [0., 0.]])
        np.testing.assert_array_equal(anchored_power(np.zeros(5), intervals), [0, 0, 0, 2, 0])
        np.testing.assert_array_equal(anchored_power(np.ones(5), intervals), intervals[:, 1])
        np.testing.assert_array_equal(anchored_power(-np.ones(5), intervals), intervals[:, 0])
        rng = np.random.default_rng(5)
        for _ in range(20):
            power = anchored_power(rng.uniform(-1, 1, 5), intervals)
            self.assertTrue(np.all(power >= intervals[:, 0]))
            self.assertTrue(np.all(power <= intervals[:, 1]))

    def test_engineered_shapes_and_empty_features(self):
        for n in range(1, 11):
            obs, _ = V2GEnv(n_slots=n).reset(seed=0, options={'date': '2026-10-03'})
            features = engineered_observation(obs)
            self.assertEqual(features['global'].shape, (18,))
            self.assertEqual(features['ev'].shape, (n, 7))
            self.assertTrue(np.isfinite(features['global']).all())
            self.assertTrue((features['ev'] == 0).all())
            self.assertTrue((environment_action(np.zeros(n + 1), obs)[1:] == 0).all())


if __name__ == '__main__':
    unittest.main()
