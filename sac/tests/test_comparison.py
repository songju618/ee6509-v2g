import unittest

from sac.comparison import compare


class ComparisonTests(unittest.TestCase):
    def row(self, day, cost, strategy):
        return dict(date=f'2026-10-{day:02d}', scenario_seed=0, strategy=strategy,
                    variable_cost_sgd=cost, total_cost_sgd=cost + 125.5, purchase_sgd=cost,
                    revenue_sgd=0., peak_increment_sgd=0., bess_wear_sgd=0., ev_wear_sgd=0.,
                    peak_grid_kw=240., bess_throughput_kwh=0., ev_throughput_kwh=0., departures=[])

    def test_positive_and_negative_paired_date_differences(self):
        sac = [self.row(day, 100., 'SAC') for day in range(15, 22)]
        worse = [self.row(day, 110., 'immediate') for day in range(15, 22)]
        report = compare(sac, worse)
        self.assertEqual(report['comparisons']['immediate']['mean_saving_sgd'], 10.)
        self.assertTrue(report['comparisons']['immediate']['significantly_better'])
        report = compare(worse, sac)
        self.assertFalse(report['comparisons']['SAC']['significantly_better'])
        self.assertLess(report['comparisons']['SAC']['mean_saving_sgd'], 0.)

    def test_fixed_dates_and_date_population_have_distinct_intervals(self):
        sac, reference = [], []
        for day in range(15, 22):
            saving = -20. if day < 19 else 50.
            for seed in range(5):
                first = self.row(day, 100., 'SAC')
                second = self.row(day, 100. + saving, 'immediate')
                first['scenario_seed'] = second['scenario_seed'] = seed
                sac.append(first)
                reference.append(second)
        result = compare(sac, reference)['comparisons']['immediate']
        self.assertTrue(result['significantly_better'])
        self.assertGreater(result['fixed_date_seed_bootstrap_95_ci_sgd'][0], 0)
        self.assertLess(result['date_cluster_bootstrap_95_ci_sgd'][0], 0)

    def test_mismatched_grids_are_rejected(self):
        with self.assertRaises(ValueError):
            compare([self.row(15, 100., 'SAC')], [self.row(16, 110., 'rule')])


if __name__ == '__main__':
    unittest.main()
