import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('isolated_local_runner', scripts/'evaluate_public_local.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.path.pop(0)


class LocalRunnerPreservationTests(unittest.TestCase):
    def test_fresh_run_is_created_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'one-attempt'
            output, scratch = runner.create_task_run(target)
            self.assertTrue(output.is_dir())
            self.assertTrue(scratch.is_dir())
            (target/'trajectory.jsonl').write_text('existing trace\n')
            with self.assertRaisesRegex(SystemExit,'preserving existing task attempt'):
                runner.create_task_run(target)
            self.assertEqual((target/'trajectory.jsonl').read_text(),'existing trace\n')

    def test_partial_attempt_without_results_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'partial'
            target.mkdir()
            (target/'trajectory.jsonl').write_text('partial generation\n')
            with self.assertRaises(SystemExit): runner.create_task_run(target)
            self.assertEqual(sorted(p.name for p in target.iterdir()),['trajectory.jsonl'])
