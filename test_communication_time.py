import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class CommunicationTimingTest(unittest.TestCase):
    def test_new_python_files_parse(self):
        for name in ("communication_time_main.py", "FedEPPU_timing.py",
                     "other/run_timing_baselines.py", "summarize_communication_time.py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_cuda_sync_brackets_timing(self):
        for name in ("FedEPPU_timing.py", "other/run_timing_baselines.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("torch.cuda.synchronize()", text)
            self.assertIn("local_train_seconds", text)
            self.assertIn("aggregation_seconds", text)
            self.assertIn("train_aggregate_seconds", text)

    def test_formal_main3_is_not_timing_entry(self):
        self.assertNotIn('["timing"]', (ROOT / "main3.py").read_text(encoding="utf-8"))

    def test_array_has_120_tasks(self):
        text = (ROOT / "run-communication-time.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-119%3", text)
        self.assertNotIn("\r\n", text)


if __name__ == "__main__":
    unittest.main()
