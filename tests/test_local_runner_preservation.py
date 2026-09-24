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
            self.assertEqual((copy/'data.json').stat().st_mode & 0o222, 0)
            (copy/'data.json').chmod(0o644)
            (copy/'data.json').write_text('{"x":3}')
            with self.assertRaisesRegex(SystemExit,'drift detected'):
                runner.require_unchanged_unit(copy,manifest['fixture'])

    def test_snapshot_blocks_writes_through_a_scratch_data_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); unit=root/'source/units/fixture'; data=unit/'environment/data'
            data.mkdir(parents=True)
            (unit/'card.toml').write_text('[task]\nid="fixture"\n')
            (data/'prices.csv').write_text('date,value\n2026-01-01,1\n')
            snapshot=root/'snapshot'
            manifest=runner.snapshot_inputs(root/'source',['fixture'],snapshot)
            frozen=snapshot/'units/fixture/environment/data/prices.csv'
            scratch=root/'scratch'; scratch.mkdir()
            (scratch/'prices.csv').symlink_to(frozen)
            with self.assertRaises(PermissionError):
                (scratch/'prices.csv').write_text('overwritten\n')
            self.assertEqual(frozen.read_text(),'date,value\n2026-01-01,1\n')
            runner.require_unchanged_unit(snapshot/'units/fixture',manifest['fixture'])

    def test_checker_staging_rewrites_only_its_readonly_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); unit=root/'unit'; checks=unit/'checks'; data=unit/'environment/data'
            checks.mkdir(parents=True); data.mkdir(parents=True)
            (unit/'card.toml').write_text('[task]\nid="fixture"\n')
            source=checks/'test_outputs.py'
            source.write_text('OUTPUT = "/app/output"\nDATA = "/app/data/x.csv"\n')
            (data/'x.csv').write_text('x\n1\n')
            runner.make_snapshot_files_read_only(unit)
            run=root/'run'; run.mkdir(); output=run/'output'; output.mkdir()
            staged=runner.stage_checker(unit,run,output)
            staged_text=staged.read_text()
            self.assertIn(str(output.resolve()),staged_text)
            self.assertIn(str(data.resolve()),staged_text)
            self.assertEqual(source.read_text(),'OUTPUT = "/app/output"\nDATA = "/app/data/x.csv"\n')
            self.assertEqual(source.stat().st_mode & 0o222,0)

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
