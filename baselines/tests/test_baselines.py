"""C's rule baselines against the README ordering on selected test cases."""
import unittest

from baselines.arrival_charge import arrival_charge_powers
from baselines.common import replay, summarize
from baselines.price_threshold import powers_from_threshold
from env import V2GEnv, load_params


# Two flat days with midday price lows, the spike day and a car-free weekend.
CASES = (("2026-10-22", 100), ("2026-10-27", 105), ("2026-10-28", 102), ("2026-10-24", 100))


class RuleBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_params()

    def scenario(self, n_slots, date, seed):
        env = V2GEnv(n_slots=n_slots, params=self.params, split="test")
        try:
            env.reset(seed=seed, options={"date": date})
            return env.export_scenario()
        finally:
            env.close()

    def test_rule_is_feasible_and_no_dearer_than_arrival_charge(self):
        for n_slots in (5, 10):
            for date, seed in CASES:
                with self.subTest(n_slots=n_slots, date=date, seed=seed):
                    scenario = self.scenario(n_slots, date, seed)
                    _, rule = replay(scenario, powers_from_threshold(scenario, self.params)[0])
                    _, arrival = replay(scenario, arrival_charge_powers(scenario, self.params))
                    self.assertLessEqual(rule["variable_cost_sgd"],
                                         arrival["variable_cost_sgd"] + 1e-6)
                    self.assertTrue(all(d["success"] for d in rule["departures"]))
                    self.assertGreaterEqual(rule["bess_final_kwh"],
                                            scenario["bess_initial_kwh"] - 1e-6)

    def test_rule_never_discharges_evs(self):
        scenario = self.scenario(10, "2026-10-28", 102)
        powers = powers_from_threshold(scenario, self.params)[0]
        self.assertGreaterEqual(powers[:, 1:].min(), 0.0)

    def test_summary_keeps_slot_count_on_car_free_days(self):
        scenario = self.scenario(5, "2026-10-24", 100)
        _, summary = replay(scenario, arrival_charge_powers(scenario, self.params))
        self.assertEqual(summary["departures"], [])
        row = summarize("2026-10-24", 100, "arrival_charge", summary, n_slots=5)
        self.assertEqual(row["n_slots"], 5)


if __name__ == "__main__":
    unittest.main()
