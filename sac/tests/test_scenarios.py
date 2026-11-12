from datetime import date as Date
import unittest

from env import V2GEnv, load_params
from env.physics import BatterySpec, feasible_interval
from env.recording import occupancy_mask
from sac.scenarios import LADDER_COUNTS, extended_scenario


class RealisticScenarioTests(unittest.TestCase):
    def test_counts_injection_and_reproducibility(self):
        for count in LADDER_COUNTS:
            scenario = extended_scenario('2026-10-03', 100, 'ladder', count)
            self.assertEqual(sum(v is not None for v in scenario.vehicles), count)
            self.assertEqual(scenario.to_dict(), extended_scenario('2026-10-03', 100, 'ladder', count).to_dict())
            V2GEnv(n_slots=10).reset(options={'scenario': scenario})

    def test_day_night_weekends_and_reachability(self):
        early = late = weekend = 0
        spec = BatterySpec.from_params(load_params()['ev'])
        for day in range(1, 15):
            for seed in range(10):
                date = f'2026-10-{day:02d}'
                scenario = extended_scenario(date, seed)
                mask = occupancy_mask(scenario)
                self.assertLessEqual(mask.sum(axis=1).max(), 10)
                early += int(mask[0].any())
                late += int(mask[46].any())
                weekend += int(Date.fromisoformat(date).weekday() >= 5 and mask.any())
                for vehicle in scenario.vehicles:
                    if vehicle is not None:
                        feasible_interval(vehicle.initial_kwh, vehicle.target_kwh,
                                          vehicle.departure - vehicle.arrival, spec)
        self.assertGreater(min(early, late, weekend), 0)


if __name__ == '__main__':
    unittest.main()
