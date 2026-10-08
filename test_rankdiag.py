import unittest

import torch

import FedEPPU_rankdiag as engine


class RankingDiagnosticTest(unittest.TestCase):
    def test_component_losses_sum_to_training_loss(self):
        logits = torch.tensor([1.0, -0.5, 0.3, -1.2], requires_grad=True)
        ranking = torch.tensor([0.8, -0.2, 0.1, -0.9], requires_grad=True)
        shared = torch.ones((4, 2), requires_grad=True)
        targets = torch.tensor([1, 1, -1, -1])
        diagnostics = {}
        total, scale, classification, rank = (
            engine.ClassificationRankingPUObjective.compute_components(
                logits, ranking, shared, targets, 0.5, diagnostics
            )
        )
        self.assertTrue(torch.allclose(total, classification + scale * rank))
        self.assertIn("classification_loss", diagnostics)
        self.assertIn("ranking_loss", diagnostics)
        self.assertIn("joint_loss", diagnostics)

    def test_gradient_norm_is_read_only(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        loss = (parameter ** 2).sum()
        norm = engine._loss_gradient_norm(loss, [parameter])
        self.assertAlmostEqual(norm, 10.0, places=6)
        self.assertIsNone(parameter.grad)
        loss.backward()
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([6.0, 8.0])))


if __name__ == "__main__":
    unittest.main()
