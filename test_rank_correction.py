import unittest

import torch

from FedEPPU_rank_correction import _apply_ranking_correction


class RankingCorrectionTest(unittest.TestCase):
    def test_negative_risk_has_opposite_absolute_gradient_and_zero_clamp_gradient(self):
        raw_abs = torch.tensor(-2.0, requires_grad=True)
        absolute = _apply_ranking_correction(raw_abs, "absolute")
        absolute.backward()
        self.assertEqual(float(absolute.item()), 2.0)
        self.assertEqual(float(raw_abs.grad.item()), -1.0)

        raw_clamp = torch.tensor(-2.0, requires_grad=True)
        clamp = _apply_ranking_correction(raw_clamp, "clamp")
        clamp.backward()
        self.assertEqual(float(clamp.item()), 0.0)
        self.assertEqual(float(raw_clamp.grad.item()), 0.0)

    def test_nonnegative_branch_is_identical(self):
        for mode in ("absolute", "clamp"):
            raw = torch.tensor(2.0, requires_grad=True)
            corrected = _apply_ranking_correction(raw, mode)
            corrected.backward()
            self.assertEqual(float(corrected.item()), 2.0)
            self.assertEqual(float(raw.grad.item()), 1.0)

    def test_unknown_correction_is_rejected(self):
        with self.assertRaises(ValueError):
            _apply_ranking_correction(torch.tensor(1.0), "unknown")


if __name__ == "__main__":
    unittest.main()
