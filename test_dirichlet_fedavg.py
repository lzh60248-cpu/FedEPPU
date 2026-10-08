import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class DirichletFedAvgTest(unittest.TestCase):
    def test_python_files_parse(self):
        for name in ("dirichlet_fedavg_nnpu.py", "summarize_dirichlet_fedavg.py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_has_four_concentrations_and_twenty_tasks(self):
        script = (ROOT / "run-dirichlet-fedavg.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-19%3", script)
        self.assertIn("ALPHAS=(0.1 0.5 1.0 2.0)", script)
        self.assertIn("SEEDS=(42 43 44 45 46)", script)

    def test_logs_required_distribution_fields(self):
        text = (ROOT / "dirichlet_fedavg_nnpu.py").read_text(encoding="utf-8")
        self.assertIn("client_priors", text)
        self.assertIn("client_sample_counts", text)
        self.assertIn("peak_to_last20_auc_drop", text)

    def test_unix_line_endings(self):
        self.assertNotIn(b"\r\n", (ROOT / "run-dirichlet-fedavg.sh").read_bytes())


if __name__ == "__main__":
    unittest.main()
