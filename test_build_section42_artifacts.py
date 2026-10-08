import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class Section42ArtifactBuilderTest(unittest.TestCase):
    def test_script_parses(self):
        name = "build_section42_artifacts.py"
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_design_contains_all_180_runs(self):
        text = (ROOT / "build_section42_artifacts.py").read_text(encoding="utf-8")
        self.assertIn('DATASETS = ("fashion_mnist", "cifar10", "chest", "kolektor_sdd2")', text)
        self.assertIn("SEEDS = (42, 43, 44, 45, 46)", text)
        self.assertIn("len(manifest) != 180", text)

    def test_runs_existing_summary_and_plotters(self):
        text = (ROOT / "build_section42_artifacts.py").read_text(encoding="utf-8")
        self.assertIn('"summarize_section42.py"', text)
        self.assertIn('"plot_section42_curves.py"', text)
        self.assertIn("subprocess.run", text)

    def test_duplicate_choice_uses_job_id(self):
        text = (ROOT / "build_section42_artifacts.py").read_text(encoding="utf-8")
        self.assertIn("job_id(p, args.input_root)", text)
        self.assertIn("candidate_count", text)


if __name__ == "__main__":
    unittest.main()
