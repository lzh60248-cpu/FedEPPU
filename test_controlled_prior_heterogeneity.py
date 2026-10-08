import unittest
import controlled_prior_heterogeneity as experiment

class ControlledPriorTest(unittest.TestCase):
    def test_configs_hold_global_prior_constant(self):
        totals={round(sum(v),12) for v in experiment.PRIOR_CONFIGS.values()}
        self.assertEqual(totals,{1.5})
    def test_configs_are_ordered_and_match_protocol(self):
        self.assertEqual(experiment.PRIOR_CONFIGS["mild"],(.30,.50,.70))
        self.assertEqual(experiment.PRIOR_CONFIGS["moderate"],(.10,.50,.90))
        self.assertEqual(experiment.PRIOR_CONFIGS["extreme"],(.01,.50,.99))
    def test_equal_client_budgets(self):
        self.assertGreater(experiment.P_PER_CLIENT,0);self.assertGreater(experiment.U_PER_CLIENT,0)

if __name__=="__main__":unittest.main()
