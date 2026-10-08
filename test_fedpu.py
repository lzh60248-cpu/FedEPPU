import math
import unittest

import torch
from torch import nn

from other.fedpu_pu import FedPUBinaryLoss, local_train_fedpu


class TinyTwoLogit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, x):
        return self.linear(x.reshape(len(x), 1))


class FedPUBinaryTest(unittest.TestCase):
    def test_binary_source_risk_does_not_cancel_correction(self):
        loss = FedPUBinaryLoss()
        outputs = torch.zeros(4, 2, requires_grad=True)
        labels = torch.tensor([1, 1, -1, -1])
        total, correction, positive, u_negative, p_negative = loss(
            outputs, labels, prior=0.25
        )
        self.assertAlmostEqual(float(positive), math.log(2), places=6)
        self.assertAlmostEqual(float(u_negative), math.log(2), places=6)
        self.assertAlmostEqual(float(p_negative), math.log(2), places=6)
        self.assertAlmostEqual(float(correction), 0.75 * math.log(2), places=6)
        self.assertAlmostEqual(float(total), 1.75 * math.log(2), places=6)

    def test_local_batches_always_contain_p_and_u(self):
        torch.manual_seed(3)
        model = TinyTwoLogit()
        x = torch.arange(20, dtype=torch.float32).reshape(20, 1, 1, 1)
        labels = torch.tensor([1, 1] + [-1] * 18)
        _, diag = local_train_fedpu(
            model, x, labels, prior=0.2, epochs=1, lr=0.001,
            batch_size=8, max_steps_per_epoch=2, seed=9,
        )
        self.assertEqual(diag["updates"], 2)
        self.assertGreaterEqual(diag["p_per_batch"], 1)
        self.assertGreaterEqual(diag["u_per_batch"], 1)
        self.assertEqual(diag["p_per_batch"] + diag["u_per_batch"], 8)

    def test_missing_pool_is_rejected(self):
        model = TinyTwoLogit()
        x = torch.zeros(4, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "both labeled P and U"):
            local_train_fedpu(model, x, torch.ones(4), prior=0.2,
                              epochs=1, lr=0.01, batch_size=2)


if __name__ == "__main__":
    unittest.main()
