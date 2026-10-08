"""Structural tests for the controlled ablations derived from Main1."""
import unittest

import numpy as np

import FedEPPU1_ablation as engine


class Main1AblationTest(unittest.TestCase):
    def test_variants_are_exactly_the_three_unique_main1_settings(self):
        expected = {
            "main1_full": (True, True, False),
            "no_ranking": (False, True, False),
            "fedavg": (True, False, False),
        }
        self.assertEqual(engine.MAIN1_ABLATION_VARIANTS, expected)
        self.assertEqual(len(set(expected.values())), len(expected))
        self.assertTrue(all(not flags[2] for flags in expected.values()))

    def test_fedavg_and_complement_mass_weights_are_the_only_aggregation_modes(self):
        counts = np.asarray([10.0, 20.0, 70.0])
        fedavg = counts / counts.sum()
        complement = engine.client_weights(counts)
        self.assertTrue(np.allclose(fedavg, np.asarray([0.1, 0.2, 0.7])))
        self.assertTrue(np.allclose(complement, np.asarray([0.1, 0.2, 0.3]) / 0.6))
        self.assertFalse(np.allclose(fedavg, complement))

    def test_balanced_counts_leave_complement_mass_equal_to_fedavg(self):
        counts = np.asarray([1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(engine.client_weights(counts), counts / counts.sum()))


if __name__ == "__main__":
    unittest.main()
