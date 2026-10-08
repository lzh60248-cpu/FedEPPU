import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class DirichletMain3Test(unittest.TestCase):
    def test_python_files_parse(self):
        for name in ("summarize_dirichlet_main3.py",):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_only_fedeppu_is_run(self):
        script = (ROOT / "run-dirichlet-main3.sh").read_text(encoding="utf-8")
        self.assertIn("srun python -u main3.py", script)
        for baseline in ("fedavg", "fedprox", "scaffold", "fedpu", "mrfl"):
            self.assertNotIn(f"--method {baseline}", script.lower())

    def test_exact_15_task_design(self):
        script = (ROOT / "run-dirichlet-main3.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-14%3", script)
        self.assertIn("DATASET=cifar10", script)
        self.assertNotIn("fashion_mnist", script)
        self.assertNotIn("kolektor_sdd2", script)
        self.assertNotIn("DATASET=chest", script)
        self.assertIn("ALPHAS=(0.1 0.5 1.0)", script)
        self.assertIn("SEEDS=(42 43 44 45 46)", script)

    def test_shell_file_uses_unix_line_endings(self):
        data = (ROOT / "run-dirichlet-main3.sh").read_bytes()
        self.assertNotIn(b"\r\n", data)


if __name__ == "__main__":
    unittest.main()
