"""Focused CPU tests for method-neutral nnPU baselines and MRFL adaptation."""
import unittest

import numpy as np
import torch
from torch import nn

from other.MRFL import (
    MRFLState,
    _minmax_columns,
    _select_reliable_pseudo_positives,
    local_train_mrfl_nnpu,
)
from other.data_protocol import _make_client, pooled_pu_prior


class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x.reshape(len(x), -1)[:, :1] * self.scale


class MRFLNnPUTest(unittest.TestCase):
    def test_protocol_keeps_pu_observations_separate_from_pn_truth(self):
        np.random.seed(3)
        x = np.arange(24, dtype=np.float32).reshape(6, 1, 2, 2)
        truth = np.asarray([1, 1, 1, -1, -1, -1], dtype=np.int32)
        client = _make_client(x, truth, n_labeled=1)
        self.assertEqual(int((client.observed_y == 1).sum()), 1)
        self.assertEqual(int((client.true_y == 1).sum()), 3)
        hidden_positive = ((client.observed_y == -1) & (client.true_y == 1))
        self.assertEqual(int(hidden_positive.sum()), 2)
        self.assertAlmostEqual(client.prior, 2.0 / 5.0)
        self.assertAlmostEqual(pooled_pu_prior([client]), client.prior)

    def test_client_division_starts_only_after_warmup(self):
        values = np.asarray([[0.1, 0.1], [0.2, 0.2], [3.0, 3.0]])
        state = MRFLState(3, warmup_rounds=2, seed=7)
        self.assertFalse(state.update_client_division(values, 0).any())
        noisy = state.update_client_division(values, 1)
        self.assertTrue(noisy[2])
        self.assertEqual(int(noisy.sum()), 1)

    def test_column_normalization_handles_constant_component(self):
        actual = _minmax_columns(np.asarray([[1.0, 2.0], [1.0, 4.0]]))
        self.assertTrue(np.allclose(actual[:, 0], 0.0))
        self.assertTrue(np.allclose(actual[:, 1], [0.0, 1.0]))

    def test_pseudo_positive_selection_is_prior_bounded(self):
        model = ScoreModel()
        x = torch.tensor([[[[-3.0]]], [[[-2.0]]], [[[2.0]]], [[[3.0]]]])
        t = torch.tensor([1, -1, -1, -1])
        adapted, mass, selected = _select_reliable_pseudo_positives(
            model, x, t, prior=1.0 / 3.0, seed=4
        )
        self.assertLessEqual(selected, 1)
        self.assertEqual(mass, 1 + selected)
        self.assertEqual(int((adapted == 1).sum()), 1 + selected)

    def test_local_training_reports_reliability_mass(self):
        model = ScoreModel()
        x = torch.tensor([[[[-3.0]]], [[[-2.0]]], [[[2.0]]], [[[3.0]]]])
        t = torch.tensor([1, -1, -1, -1])
        _, diag = local_train_mrfl_nnpu(
            model, x, t, prior=0.25, epochs=1, lr=0.01,
            batch_size=4, is_noisy=True, seed=5,
        )
        self.assertTrue(diag["noisy_client"])
        self.assertGreaterEqual(diag["aggregation_mass"], 1)
        self.assertEqual(diag["mrfl_stage"], "robust")


if __name__ == "__main__":
    unittest.main()
