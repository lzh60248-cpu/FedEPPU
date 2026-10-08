import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ProxySizeMain3Test(unittest.TestCase):
    def test_python_files_parse(self):
        for name in ("main3.py", "datasets/dataset.py", "summarize_plot_proxy_size_main3.py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_exact_twenty_task_design(self):
        script = (ROOT / "run-proxy-size-main3.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-19%3", script)
        self.assertIn("PROXY_SIZES=(128 256 512 1024)", script)
        self.assertIn("SEEDS=(42 43 44 45 46)", script)
        self.assertIn("PROXY_POOL_SIZE=1024", script)

    def test_fixed_client_partition_protocol(self):
        script = (ROOT / "run-proxy-size-main3.sh").read_text(encoding="utf-8")
        self.assertIn('--n_proxy "$N_PROXY"', script)
        self.assertIn('--proxy_pool_size "$PROXY_POOL_SIZE"', script)
        data = (ROOT / "datasets/dataset.py").read_text(encoding="utf-8")
        self.assertIn("perm[reserved_proxy_count:]", data)
        summary = (ROOT / "summarize_plot_proxy_size_main3.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Client partition changed with proxy size", summary)
        self.assertIn("across all retry job folders", summary)
        self.assertIn("candidate_count", summary)

    def test_shell_uses_unix_line_endings(self):
        path = ROOT / "run-proxy-size-main3.sh"
        self.assertNotIn(b"\r\n", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
