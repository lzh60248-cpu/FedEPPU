"""Checks for the single-seed, prior-coupled sample-heterogeneity protocol."""

import unittest

import numpy as np

import sample_prior_coupled_main3 as experiment


class PriorCoupledSampleTest(unittest.TestCase):
    def test_every_configuration_preserves_global_unlabeled_pools(self):
        for ratio in experiment.SIZE_RATIOS.values():
            unlabeled = experiment.allocate(
                experiment.TOTAL_UNLABELED_POSITIVE + experiment.TOTAL_UNLABELED_NEGATIVE,
                ratio,
            )
            positive = experiment.coupled_unlabeled_positive_counts(unlabeled)
            negative = unlabeled - positive
            self.assertEqual(int(positive.sum()), experiment.TOTAL_UNLABELED_POSITIVE)
            self.assertEqual(int(negative.sum()), experiment.TOTAL_UNLABELED_NEGATIVE)
            self.assertEqual(int((positive + negative).sum()), int(unlabeled.sum()))

    def test_largest_client_has_target_prior_and_smaller_clients_supply_remaining_evidence(self):
        for ratio in experiment.SIZE_RATIOS.values():
            unlabeled = experiment.allocate(
                experiment.TOTAL_UNLABELED_POSITIVE + experiment.TOTAL_UNLABELED_NEGATIVE,
                ratio,
            )
            positive = experiment.coupled_unlabeled_positive_counts(unlabeled)
            priors = positive / unlabeled
            self.assertAlmostEqual(priors[0], experiment.LARGEST_CLIENT_PRIOR, places=4)
            self.assertLess(priors[1], priors[0])
            self.assertLess(priors[2], priors[0])
            self.assertAlmostEqual(
                float(positive.sum() / unlabeled.sum()),
                experiment.TOTAL_UNLABELED_POSITIVE /
                (experiment.TOTAL_UNLABELED_POSITIVE + experiment.TOTAL_UNLABELED_NEGATIVE),
                places=12,
            )

    def test_task_grid_contains_nine_unique_runs(self):
        configs = tuple(experiment.SIZE_RATIOS)
        methods = experiment.METHODS
        runs = [(configs[task // 3], methods[task % 3], 44) for task in range(9)]
        self.assertEqual(len(runs), 9)
        self.assertEqual(len(set(runs)), 9)


if __name__ == "__main__":
    unittest.main()