import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class DirichletComparisonMain3Test(unittest.TestCase):
    def test_python_files_parse(self):
        for name in ("summarize_plot_dirichlet_comparison_main3.py",):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_has_two_methods_four_alphas_and_forty_tasks(self):
        script = (ROOT / "run-dirichlet-comparison-main3.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-39%3", script)
        self.assertIn("ALPHAS=(0.1 0.5 1.0 2.0)", script)
        self.assertIn("METHODS=(fedavg_nnpu fedeppu)", script)
        self.assertIn("SEEDS=(42 43 44 45 46)", script)

    def test_summary_records_table_four_fields(self):
        text = (ROOT / "summarize_plot_dirichlet_comparison_main3.py").read_text(encoding="utf-8")
        for field in ("actual_prior_range", "actual_sample_range", "last20_auc_sd", "peak_drop"):
            self.assertIn(field, text)

    def test_shell_uses_unix_line_endings(self):
        self.assertNotIn(b"\r\n", (ROOT / "run-dirichlet-comparison-main3.sh").read_bytes())


if __name__ == "__main__":
    unittest.main()
