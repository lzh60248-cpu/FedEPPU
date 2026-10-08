import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class DirichletArtifactTest(unittest.TestCase):
    def test_builder_parses(self):
        ast.parse(
            (ROOT / "build_dirichlet_comparison_artifacts.py").read_text(encoding="utf-8"),
            filename="build_dirichlet_comparison_artifacts.py",
        )

    def test_builder_merges_all_expected_runs(self):
        text = (ROOT / "build_dirichlet_comparison_artifacts.py").read_text(encoding="utf-8")
        self.assertIn('CONFIGS = ("alpha0p1", "alpha0p5", "alpha1p0", "alpha2p0")', text)
        self.assertIn('METHODS = ("fedavg_nnpu", "fedeppu")', text)
        self.assertIn("found {len(manifest)}/40 results", text)

    def test_one_non_array_plot_job(self):
        text = (ROOT / "run-dirichlet-comparison-artifacts.sh").read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --array", text)
        self.assertIn("build_dirichlet_comparison_artifacts.py", text)
        self.assertNotIn(b"\r\n", (ROOT / "run-dirichlet-comparison-artifacts.sh").read_bytes())


if __name__ == "__main__":
    unittest.main()
