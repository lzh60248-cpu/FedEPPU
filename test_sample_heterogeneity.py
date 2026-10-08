import unittest

import numpy as np

from FedEPPU_ablation import ComplementMassAggregation, linear_ranking_support
from sample_heterogeneity import (
    SIZE_RATIOS, TOTAL_LABELED, TOTAL_UNLABELED_NEGATIVE,
    TOTAL_UNLABELED_POSITIVE, _allocate,
)


class SampleHeterogeneityTest(unittest.TestCase):
    def test_counts_follow_requested_ratios_and_preserve_totals(self):
        totals = (TOTAL_LABELED, TOTAL_UNLABELED_POSITIVE,
                  TOTAL_UNLABELED_NEGATIVE)
        for ratio in SIZE_RATIOS.values():
            allocations = [_allocate(total, ratio) for total in totals]
            self.assertTrue(all(int(x.sum()) == total
                                for x, total in zip(allocations, totals)))
            client_counts = sum(allocations)
            scale = client_counts / np.asarray(ratio)
            self.assertTrue(np.all(scale == scale[0]))

    def test_client_unlabeled_priors_are_identical(self):
        expected = TOTAL_UNLABELED_POSITIVE / (
            TOTAL_UNLABELED_POSITIVE + TOTAL_UNLABELED_NEGATIVE
        )
        for ratio in SIZE_RATIOS.values():
            up = _allocate(TOTAL_UNLABELED_POSITIVE, ratio)
            un = _allocate(TOTAL_UNLABELED_NEGATIVE, ratio)
            self.assertTrue(np.allclose(up/(up+un), expected))

    def test_complement_mass_only_changes_absolute_majority_cases(self):
        equal = ComplementMassAggregation.weights(np.asarray((1, 1, 1), float))
        moderate = ComplementMassAggregation.weights(np.asarray((6, 3, 1), float))
        extreme = ComplementMassAggregation.weights(np.asarray((8, 1, 1), float))
        self.assertTrue(np.allclose(equal, (1/3, 1/3, 1/3)))
        self.assertTrue(np.allclose(moderate, (0.5, 0.375, 0.125)))
        self.assertTrue(np.allclose(extreme, (0.5, 0.25, 0.25)))

    def test_equal_fixed_priors_make_support_correction_neutral(self):
        prior = TOTAL_UNLABELED_POSITIVE / (
            TOTAL_UNLABELED_POSITIVE + TOTAL_UNLABELED_NEGATIVE
        )
        supports = np.asarray([linear_ranking_support(prior, 192)] * 3)
        self.assertTrue(np.allclose(supports, 1.0))
        base = np.asarray((0.6, 0.3, 0.1))
        actual = ComplementMassAggregation.support_aware_weights(base, supports)
        self.assertTrue(np.allclose(actual, base))


if __name__ == "__main__":
    unittest.main()
