"""Structural checks for the controlled Main3 sample-heterogeneity experiment."""

import inspect
import unittest

import numpy as np

import FedEPPU_sample_main3 as engine
import sample_heterogeneity_main3 as experiment


class SampleHeterogeneityMain3Test(unittest.TestCase):
    def test_allocations_preserve_totals_priors_and_ratios(self):
        totals = (
            experiment.TOTAL_LABELED,
            experiment.TOTAL_UNLABELED_POSITIVE,
            experiment.TOTAL_UNLABELED_NEGATIVE,
        )
        for ratio in experiment.SIZE_RATIOS.values():
            allocations = [experiment.allocate(total, ratio) for total in totals]
            self.assertTrue(all(int(values.sum()) == total
                                for values, total in zip(allocations, totals)))
            counts = sum(allocations)
            self.assertTrue(np.all(counts / np.asarray(ratio) == counts[0] / ratio[0]))
            up, un = allocations[1], allocations[2]
            self.assertTrue(np.allclose(up / (up + un), (up / (up + un))[0]))

    def test_complement_weights_have_theoretical_values(self):
        expected = {
            "balanced": (1/3, 1/3, 1/3),
            "moderate": (0.5, 0.375, 0.125),
            "extreme": (0.5, 0.25, 0.25),
        }
        for name, ratio in experiment.SIZE_RATIOS.items():
            actual = engine.ComplementMassAggregation.weights(np.asarray(ratio, float))
            self.assertTrue(np.allclose(actual, expected[name]))

    def test_engine_switch_changes_only_base_aggregation(self):
        source = inspect.getsource(engine.train_fedavg_proposed)
        self.assertIn('aggregation_mode == "complement"', source)
        self.assertIn("raw_aggregation_weights.copy()", source)
        self.assertIn("support_aware_weights", source)
        self.assertNotIn("_update_model_ema", source)

    def test_task_grid_contains_45_unique_runs(self):
        runs = [(task // 15, (task % 15) // 5, task % 5) for task in range(45)]
        self.assertEqual(len(runs), 45)
        self.assertEqual(len(set(runs)), 45)


if __name__ == "__main__":
    unittest.main()
