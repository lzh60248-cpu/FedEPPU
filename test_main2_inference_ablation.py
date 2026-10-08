"""Structural tests for Main2 inference-pipeline ablation isolation."""
import unittest

import FedEPPU2_inference_ablation as engine


class InferenceAblationTest(unittest.TestCase):
    def test_variants_change_only_requested_inference_switches(self):
        expected = {
            "full": (True, True),
            "last_raw": (False, True),
            "zero_threshold": (True, False),
            "last_raw_zero": (False, False),
        }
        self.assertEqual(engine.MAIN2_INFERENCE_VARIANTS, expected)

    def test_zero_threshold_variants_disable_calibration(self):
        for name in ("zero_threshold", "last_raw_zero"):
            self.assertFalse(engine.MAIN2_INFERENCE_VARIANTS[name][1])
        for name in ("full", "last_raw"):
            self.assertTrue(engine.MAIN2_INFERENCE_VARIANTS[name][1])

    def test_model_ema_is_absent_from_every_variant(self):
        self.assertFalse(hasattr(engine, "_update_model_ema"))
        self.assertFalse(hasattr(engine, "EMASelectionQuantileInference"))
        self.assertTrue(hasattr(engine, "ProxyCheckpointInference"))


if __name__ == "__main__":
    unittest.main()
