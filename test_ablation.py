"""Structural tests for isolated FedEPPU ablation switches."""
import unittest

import numpy as np
import torch

import FedEPPU_ablation as engine


class AblationTest(unittest.TestCase):
    def test_ranking_switch_changes_only_ranking_term(self):
        logits = torch.randn(16, requires_grad=True)
        ranking = torch.randn(16, requires_grad=True)
        features = torch.randn(16, 4, requires_grad=True)
        labels = torch.tensor([1] * 4 + [-1] * 12)
        full, full_scale = engine.ClassificationRankingPUObjective.compute(
            logits, ranking, features, labels, 0.4, use_ranking=True
        )
        diagnostics = {}
        classification, off_scale = engine.ClassificationRankingPUObjective.compute(
            logits, ranking, features, labels, 0.4,
            diagnostics=diagnostics, use_ranking=False
        )
        self.assertEqual(float(full_scale.item()), 1.0)
        self.assertEqual(float(off_scale.item()), 0.0)
        self.assertFalse(torch.equal(full, classification))
        self.assertEqual(diagnostics["ranking_support"], 1.0)

    def test_aggregation_switches_are_distinct_for_majority_and_support(self):
        counts = np.asarray([364.0, 3786.0, 44826.0])
        raw = counts / counts.sum()
        complement = engine.ComplementMassAggregation.weights(counts)
        self.assertFalse(np.allclose(raw, complement))
        support = np.asarray([0.2, 1.0, 1.0])
        adjusted = engine.ComplementMassAggregation.support_aware_weights(
            complement, support
        )
        self.assertFalse(np.allclose(adjusted, complement))
        self.assertAlmostEqual(float(adjusted.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
