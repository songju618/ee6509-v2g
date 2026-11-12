from dataclasses import replace
import unittest

import numpy as np

from env import V2GEnv, Vehicle
from sac.building import building_summary
from sac.evaluate import rollout
from sac.offline import solve_offline
from sac.scenarios import extended_scenario


class OfflineReferenceTests(unittest.TestCase):
    def test_positive_price_lp_replays_and_dominates_immediate(self):
        for n in (1, 5, 10):
            for date in ('2026-10-01', '2026-10-03'):
                scenario = extended_scenario(date, 0, n_slots=n)
                infos, summary, exported, powers = solve_offline(scenario)
                _, immediate, _, _ = rollout(V2GEnv(n_slots=n), date, 0, reference='immediate', scenario=scenario)
                self.assertEqual(len(infos), 48)
                self.assertEqual(powers.shape, (48, n + 1))
                self.assertLess(summary['objective_replay_error_sgd'], .01)
                self.assertLessEqual(summary['variable_cost_sgd'], immediate['variable_cost_sgd'] + .01)
                self.assertTrue(all(d['success'] for d in summary['departures']))
                self.assertGreaterEqual(summary['bess_final_kwh'], exported['bess_initial_kwh'] - 1e-9)

    def test_one_step_reference_matches_independent_hand_calculation(self):
        base = extended_scenario('2026-10-01', 0, n_slots=1)
        scenario = replace(base, truth=np.array([[.2], [200.], [0.]]), noise=base.noise[:, :2],
                           vehicles=(Vehicle(0, 1, 20., 22.),))
        _, summary, _, _ = solve_offline(scenario)
        power = 2 / (.9 * .5)
        expected = (200 + power) * .23 * .5 + .03118 * power * .5
        self.assertAlmostEqual(summary['objective_lower_bound_sgd'], expected, places=8)

    def test_negative_prices_use_exact_device_sign_milp(self):
        base = extended_scenario('2026-10-01', 0, n_slots=1)
        truth = base.truth[:, :6].copy()
        truth[0] = [-.2, -.1, .2, .5, .2, .1]
        scenario = replace(base, truth=truth, noise=base.noise[:, :7], vehicles=(None,))
        _, summary, _, _ = solve_offline(scenario)
        self.assertEqual(summary['solver_method'], 'MILP')
        self.assertLess(summary['objective_replay_error_sgd'], .01)

    def test_building_only_is_device_free_counterfactual(self):
        scenario = extended_scenario('2026-10-01', 0, n_slots=5).to_dict()
        first = building_summary(scenario)
        scenario['vehicles'] = [None] * 5
        env = V2GEnv(n_slots=5)
        env.reset(options={'scenario': scenario})
        infos = [env.step_power(np.zeros(6))[4] for _ in range(48)]
        self.assertAlmostEqual(first['variable_cost_sgd'],
                               infos[-1]['episode_summary']['variable_cost_sgd'], places=8)


if __name__ == '__main__':
    unittest.main()
