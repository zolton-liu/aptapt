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
    def test_input_snapshot_preserves_bytes_and_detects_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); unit=root/'source/units/fixture'; unit.mkdir(parents=True)
            (unit/'card.toml').write_text('[task]\nid="fixture"\n')
            (unit/'data.json').write_text('{"x":1}')
            snapshot=root/'snapshot'
            manifest=runner.snapshot_inputs(root/'source',['fixture'],snapshot)
            copy=snapshot/'units/fixture'
            runner.require_unchanged_unit(copy,manifest['fixture'])
            (unit/'data.json').write_text('{"x":2}')
            self.assertEqual((copy/'data.json').read_text(),'{"x":1}')
            (copy/'data.json').write_text('{"x":3}')
            with self.assertRaisesRegex(SystemExit,'drift detected'):
                runner.require_unchanged_unit(copy,manifest['fixture'])

    def test_snapshot_rejects_symlink_and_detects_added_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); unit=root/'unit'; unit.mkdir()
            (unit/'x').write_text('x')
            manifest=runner.unit_hashes(unit)
            (unit/'y').write_text('y')
            with self.assertRaises(SystemExit):runner.require_unchanged_unit(unit,manifest)
            (unit/'link').symlink_to(unit/'x')
            with self.assertRaisesRegex(SystemExit,'symlinks'):runner.unit_hashes(unit)

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
