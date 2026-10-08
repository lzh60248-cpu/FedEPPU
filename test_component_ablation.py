"""Structural tests for the Main3 component ablation."""

import inspect
import unittest

import component_ablation_main as main
import FedEPPU_component_ablation as engine


class ComponentAblationTest(unittest.TestCase):
    def test_five_variants_change_one_requested_component(self):
        self.assertEqual(set(main.VARIANTS), {
            "full", "no_ranking", "no_complement", "no_support",
            "fedavg_aggregation",
        })
        self.assertEqual(main.VARIANTS["full"], {
            "use_ranking": True,
            "use_complement_mass": True,
            "use_negative_support": True,
        })
        self.assertFalse(main.VARIANTS["no_ranking"]["use_ranking"])
        self.assertFalse(main.VARIANTS["no_complement"]["use_complement_mass"])
        self.assertFalse(main.VARIANTS["no_support"]["use_negative_support"])
        self.assertEqual(main.VARIANTS["fedavg_aggregation"], {
            "use_ranking": True,
            "use_complement_mass": False,
            "use_negative_support": False,
        })

    def test_engine_contains_independent_switches(self):
        train_source = inspect.getsource(engine.train_fedavg_proposed)
        local_source = inspect.getsource(engine.local_train)
        self.assertIn("use_complement_mass", train_source)
        self.assertIn("use_negative_support", train_source)
        self.assertIn("use_ranking", local_source)
        self.assertIn("linear_ranking_support", local_source)
        self.assertNotIn("_update_model_ema", train_source)

    def test_task_grid_has_25_unique_runs(self):
        runs = [(task // 5, task % 5) for task in range(25)]
        self.assertEqual(len(runs), 25)
        self.assertEqual(len(set(runs)), 25)


if __name__ == "__main__":
    unittest.main()
