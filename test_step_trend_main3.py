import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class StepTrendMain3Test(unittest.TestCase):
    def test_summary_parses(self):
        name = "summarize_plot_step_trend_main3.py"
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_exact_25_task_design(self):
        script = (ROOT / "run-step-trend-main3.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-24%3", script)
        self.assertIn("STEP_CAPS=(5 10 15 20 30)", script)
        self.assertIn("SEEDS=(42 43 44 45 46)", script)
        self.assertIn("--dataset cifar10", script)
        self.assertIn("srun python -u main3.py", script)

    def test_actual_steps_are_reported(self):
        text = (ROOT / "summarize_plot_step_trend_main3.py").read_text(encoding="utf-8")
        self.assertIn('"actual_steps_per_epoch"', text)
        self.assertIn("mean_actual_steps_per_epoch", text)

    def test_formal_files_are_referenced_not_replaced(self):
        script = (ROOT / "run-step-trend-main3.sh").read_text(encoding="utf-8")
        self.assertIn("main3.py FedEPPU3.py", script)

    def test_shell_uses_unix_line_endings(self):
        self.assertNotIn(b"\r\n", (ROOT / "run-step-trend-main3.sh").read_bytes())


if __name__ == "__main__":
    unittest.main()
