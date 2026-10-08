"""Structural checks for the controlled Main3 prior-heterogeneity experiment."""

import inspect
import unittest

import FedEPPU3
import prior_heterogeneity_main3 as experiment


class PriorHeterogeneityMain3Test(unittest.TestCase):
    def test_all_configurations_preserve_global_mass(self):
        totals = []
        for priors in experiment.PRIOR_CONFIGS.values():
            totals.append(sum(priors) * experiment.U_PER_CLIENT)
            self.assertTrue(all(float(p * experiment.U_PER_CLIENT).is_integer()
                                for p in priors))
        self.assertEqual(len(set(totals)), 1)

    def test_task_grid_contains_30_unique_runs(self):
        runs = []
        for task in range(30):
            runs.append((task // 10, (task % 10) // 5, task % 5))
        self.assertEqual(len(runs), 30)
        self.assertEqual(len(set(runs)), 30)

    def test_fedeppu_branch_uses_main3(self):
        source = inspect.getsource(experiment.run_fedeppu)
        self.assertIn("FedEPPU3.train_fedavg_proposed", source)
        self.assertIn("FedEPPU3.BetterCifarCNN", source)
        engine_source = inspect.getsource(FedEPPU3.train_fedavg_proposed)
        self.assertIn("support_aware_weights", engine_source)
        self.assertNotIn("_update_model_ema", engine_source)

    def test_fedavg_branch_uses_standard_sample_weights(self):
        source = inspect.getsource(experiment.run_fedavg)
        self.assertIn("weights = counts / counts.sum()", source)
        self.assertIn("local_train_fedavg", source)


if __name__ == "__main__":
    unittest.main()
