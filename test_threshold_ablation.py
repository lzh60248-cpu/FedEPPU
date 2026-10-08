"""Tests for paired calibrated-versus-zero threshold evaluation."""

import inspect
import unittest

import numpy as np
import torch

import FedEPPU_threshold_ablation as engine


class ThresholdAblationTest(unittest.TestCase):
    def test_shared_logits_keep_auc_identical(self):
        logits = np.asarray([-1.0, -0.2, 0.1, 1.2])
        labels = torch.tensor([-1, 1, -1, 1])
        calibrated = engine.metrics_from_logits(logits, labels, fixed_threshold=-0.3)
        zero = engine.metrics_from_logits(logits, labels, fixed_threshold=0.0)
        self.assertAlmostEqual(calibrated[2], zero[2])
        self.assertNotEqual(calibrated[6], zero[6])

    def test_training_loop_records_both_thresholds_from_shared_logits(self):
        source = inspect.getsource(engine.train_fedavg_proposed)
        self.assertIn("test_logits = _infer_logits", source)
        self.assertIn("fixed_threshold=reporting_threshold", source)
        self.assertIn("fixed_threshold=0.0", source)
        self.assertIn('history["selected_zero"]', source)
        self.assertNotIn("_update_model_ema", source)

    def test_zero_history_contains_required_metrics(self):
        source = inspect.getsource(engine.train_fedavg_proposed)
        for key in ("zero_acc", "zero_f1", "zero_bacc", "zero_auc"):
            self.assertIn(key, source)


if __name__ == "__main__":
    unittest.main()
