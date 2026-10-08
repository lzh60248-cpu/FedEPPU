import unittest

import torch

import FedEPPU_rankgate as engine


class RankingGateTest(unittest.TestCase):
    def _compute(self, gate, prior=0.9995):
        logits = torch.tensor([1.0, 0.5, -0.2, -1.0], requires_grad=True)
        ranking = torch.tensor([0.8, 0.4, -0.1, -0.7], requires_grad=True)
        shared = torch.ones((4, 2), requires_grad=True)
        targets = torch.tensor([1, 1, -1, -1])
        return engine.ClassificationRankingPUObjective.compute_components(
            logits, ranking, shared, targets, prior, {}, ranking_gate=gate
        )

    def test_original_uses_unit_scale(self):
        total, scale, classification, ranking = self._compute("original")
        self.assertEqual(float(scale), 1.0)
        self.assertTrue(torch.allclose(total, classification + ranking))

    def test_continuous_uses_expected_negative_support(self):
        total, scale, classification, ranking = self._compute("continuous")
        self.assertAlmostEqual(float(scale), 0.001, places=6)
        self.assertTrue(torch.allclose(total, classification + scale * ranking))

    def test_hard_gate_disables_insufficient_support(self):
        total, scale, classification, _ = self._compute("hard")
        self.assertEqual(float(scale), 0.0)
        self.assertTrue(torch.allclose(total, classification))

    def test_invalid_gate_is_rejected(self):
        with self.assertRaises(ValueError):
            self._compute("invalid")


if __name__ == "__main__":
    unittest.main()
