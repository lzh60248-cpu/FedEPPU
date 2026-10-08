import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ClientCountMain3Test(unittest.TestCase):
    def test_summary_parses(self):
        name = "summarize_plot_client_count_main3.py"
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_exact_twenty_task_design(self):
        script = (ROOT / "run-client-count-main3.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-19%3", script)
        self.assertIn("CLIENT_COUNTS=(3 6 9 12)", script)
        self.assertIn("SEEDS=(42 43 44 45 46)", script)
        self.assertIn('TASK_ID / 5', script)
        self.assertIn('TASK_ID % 5', script)

    def test_protocol_is_fixed(self):
        script = (ROOT / "run-client-count-main3.sh").read_text(encoding="utf-8")
        for text in (
            "--dataset cifar10", "--rounds 200", "--alpha 0.1",
            "--batch_size 256", "--max_steps_per_epoch 15",
            "--local_epochs 6", 'srun python -u main3.py',
        ):
            self.assertIn(text, script)

    def test_summary_requires_complete_runs(self):
        text = (ROOT / "summarize_plot_client_count_main3.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("exactly 20 valid runs", text)
        self.assertIn('curve.size != 200', text)
        self.assertIn('history.get("selected")', text)
        self.assertIn("newest complete legacy job directory", text)

    def test_shell_uses_unix_line_endings(self):
        path = ROOT / "run-client-count-main3.sh"
        self.assertNotIn(b"\r\n", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
