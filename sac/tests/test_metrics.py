import unittest

from sac.metrics import percentage_metrics, price_regime_definition, ratio


class PercentageTests(unittest.TestCase):
    def test_percentages_allow_storage_to_offset_building_cost(self):
        values = percentage_metrics(90., 110., 150., 80., 100.)
        self.assertAlmostEqual(values['optimality_gap_closure_from_immediate']['percent'], 200 / 3)
        self.assertAlmostEqual(values['optimality_gap_closure_from_random']['percent'], 600 / 7)
        self.assertEqual(values['net_ev_storage_cost_reduction']['percent'], 200.)

    def test_zero_denominators_are_explicit(self):
        value = ratio(10., 0.)
        self.assertIsNone(value['percent'])
        self.assertEqual(value['status'], 'zero_denominator')

    def test_price_regime_threshold_only_fits_training_dates(self):
        definition = price_regime_definition()
        self.assertEqual(definition['fitted_split'], 'train')
        self.assertGreater(definition['threshold_sgd_per_kwh'], 0)


if __name__ == '__main__':
    unittest.main()
