"""Structural tests for the Section 4.2 experiment and artifact pipeline."""

import inspect
import unittest
from pathlib import Path

import main3
import FedEPPU3
from other.centralized import train_centralized_nnpu, train_centralized_pn
from other.local_nnpu import train_local_models
from other.run_section42_baselines import METHODS
import plot_section42_curves as plotter
import summarize_section42 as summary


class Section42Test(unittest.TestCase):
    def test_task_grid_has_180_unique_runs(self):
        decoded = []
        for task_id in range(180):
            dataset_index = task_id // 45
            within_dataset = task_id % 45
            method_index = within_dataset // 5
            seed_index = within_dataset % 5
            decoded.append((dataset_index, method_index, seed_index))
        self.assertEqual(len(decoded), 180)
        self.assertEqual(len(set(decoded)), 180)
        self.assertEqual(decoded[0], (0, 0, 0))
        self.assertEqual(decoded[-1], (3, 8, 4))

    def test_method_and_dataset_registries_match(self):
        self.assertEqual(tuple(METHODS) + ("fedeppu",), summary.METHODS)
        self.assertEqual(summary.METHODS, plotter.METHODS)
        self.assertEqual(summary.DATASETS, plotter.DATASETS)
        self.assertEqual(summary.SEEDS, plotter.SEEDS)

    def test_read_only_round_callbacks_are_available(self):
        self.assertIn("round_callback", inspect.signature(train_centralized_pn).parameters)
        self.assertIn("round_callback", inspect.signature(train_centralized_nnpu).parameters)
        self.assertIn("round_callback", inspect.signature(train_local_models).parameters)

    def test_fedeppu_is_exactly_main3(self):
        self.assertIs(main3.train_fedavg_proposed, FedEPPU3.train_fedavg_proposed)
        source = inspect.getsource(FedEPPU3.train_fedavg_proposed)
        self.assertIn("support_aware_weights", source)
        self.assertNotIn("_update_model_ema", source)

    def test_hpc_script_declares_full_array(self):
        source = Path("run-section42.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-179%3", source)
        self.assertIn("python -u main3.py", source)
        self.assertIn("other.run_section42_baselines", source)


if __name__ == "__main__":
    unittest.main()
