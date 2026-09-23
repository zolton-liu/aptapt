from __future__ import annotations

import unittest

from qfa_agent.protocol import ProtocolError, compact_action, parse_action
from qfa_agent.types import Action


class ProtocolTests(unittest.TestCase):
    def test_parses_plain_object(self) -> None:
        action = parse_action('{"tool":"list_files","arguments":{"area":"input"}}')
        self.assertEqual(action.tool, "list_files")
        self.assertEqual(action.arguments["area"], "input")

    def test_extracts_object_from_fence(self) -> None:
        action = parse_action('```json\n{"tool":"finish","arguments":{}}\n```')
        self.assertEqual(action.tool, "finish")

    def test_accepts_flattened_tool_arguments(self) -> None:
        action = parse_action(
            '{"tool":"list_files","area":"input","path":"environment/data"}'
        )
        self.assertEqual(action.tool, "list_files")
        self.assertEqual(
            action.arguments,
            {"area": "input", "path": "environment/data"},
        )

    def test_rejects_missing_tool(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_action('{"arguments":{}}')

    def test_compact_action_preserves_small_replacements(self) -> None:
        compacted = compact_action(
            Action(
                tool="replace_text",
                arguments={"path": "scratch/solve.py", "old": "before", "new": "after"},
            )
        )
        self.assertIn('"old": "before"', compacted)
        self.assertIn('"new": "after"', compacted)

    def test_compact_action_records_source_digest_but_keeps_moderate_replacements(self) -> None:
        compacted = compact_action(
            Action(
                tool="write_file",
                arguments={"content": "x" * 20, "old": "y" * 501},
            )
        )
        self.assertNotIn('"content":', compacted)
        self.assertIn('"content_record"', compacted)
        self.assertIn('"characters": 20', compacted)
        self.assertIn("not source text", compacted)
        self.assertIn("y" * 501, compacted)

    def test_compacted_write_record_is_recovered_as_source_read(self) -> None:
        action = parse_action(
            {
                "tool": "write_file",
                "arguments": {
                    "path": "scratch/solve.py",
                    "content_record": {"omitted": True, "characters": 9000},
                    "overwrite": True,
                },
            }
        )
        self.assertEqual(action.tool, "read_file")
        self.assertEqual(action.arguments["path"], "scratch/solve.py")

    def test_compacted_replace_record_is_recovered_as_source_read(self) -> None:
        action = parse_action(
            {
                "tool": "replace_text",
                "arguments": {
                    "path": "scratch/solve.py",
                    "old_record": {"omitted": True},
                    "new": "replacement",
                },
            }
        )
        self.assertEqual(action.tool, "read_file")

    def test_compacted_replace_lines_record_is_recovered_as_source_read(self) -> None:
        action = parse_action(
            {
                "tool": "replace_lines",
                "arguments": {
                    "path": "scratch/solve.py",
                    "start_line": 8,
                    "end_line": 10,
                    "content_record": {"omitted": True},
                },
            }
        )
        self.assertEqual(action.tool, "read_file")
        self.assertEqual(action.arguments["start_line"], 8)
        self.assertEqual(action.arguments["end_line"], 10)


if __name__ == "__main__":
    unittest.main()
