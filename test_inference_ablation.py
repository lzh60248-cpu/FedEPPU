"""Structural tests for inference-pipeline ablation isolation."""
import unittest

import FedEPPU_inference_ablation as engine


class InferenceAblationTest(unittest.TestCase):
    def test_variants_change_only_requested_inference_switches(self):
        expected = {
            "full": (True, True, True),
            "no_ema": (False, True, True),
            "last_ema": (True, False, True),
            "zero_threshold": (True, True, False),
            "last_raw_zero": (False, False, False),
        }
        self.assertEqual(engine.INFERENCE_VARIANTS, expected)

    def test_zero_threshold_variants_disable_calibration(self):
        for name in ("zero_threshold", "last_raw_zero"):
            self.assertFalse(engine.INFERENCE_VARIANTS[name][2])
        for name in ("full", "no_ema", "last_ema"):
            self.assertTrue(engine.INFERENCE_VARIANTS[name][2])


if __name__ == "__main__":
    unittest.main()
