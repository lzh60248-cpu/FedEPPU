import unittest
import torch

from FedEPPU_rho_sensitivity import _positive_batch_count, sample_pu_indices


class PositiveAllocationRatioTest(unittest.TestCase):
    def test_expected_batch_counts(self):
        expected = {0.125: 32, 0.25: 64, 0.375: 96, 0.5: 128}
        for rho, count in expected.items():
            self.assertEqual(_positive_batch_count(256, rho), count)

    def test_sampler_preserves_pool_membership(self):
        positive = torch.arange(0, 10)
        unlabeled = torch.arange(10, 30)
        for rho in (0.125, 0.25, 0.375, 0.5):
            batch = sample_pu_indices(positive, unlabeled, 256, rho)
            n_positive = _positive_batch_count(256, rho)
            self.assertEqual(batch.numel(), 256)
            self.assertTrue(torch.all(batch[:n_positive] < 10))
            self.assertTrue(torch.all(batch[n_positive:] >= 10))

    def test_invalid_ratios_are_rejected(self):
        for rho in (0.0, 1.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                _positive_batch_count(256, rho)


if __name__ == "__main__":
    unittest.main()
