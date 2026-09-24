"""Parse one structured action from a model response."""

from __future__ import annotations

import json
import hashlib
from typing import Any

from .types import Action


class ProtocolError(ValueError):
    pass


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ProtocolError("model response contains no JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ProtocolError("model response contains an incomplete JSON object")


def parse_action(response: str | dict[str, Any]) -> Action:
    if isinstance(response, dict):
        payload = response
    else:
        try:
            payload = json.loads(_first_json_object(response.strip()))
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid JSON action: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ProtocolError("action must be a JSON object")
    tool = payload.get("tool")
    # Small local/open models often flatten tool parameters into the action
    # object even when the prompt asks for an ``arguments`` wrapper.  Accept
    # that equivalent representation and leave per-tool validation to the
    # router.  When the wrapper is present it remains authoritative, so a
    # malformed mixed response cannot silently override nested arguments.
    if "arguments" in payload:
        arguments = payload["arguments"]
    else:
        arguments = {key: value for key, value in payload.items() if key != "tool"}
    if not isinstance(tool, str) or not tool.strip():
        raise ProtocolError("action.tool must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ProtocolError("action.arguments must be an object")
    # A small model may copy the compacted history record back as if it were
    # source.  The record intentionally contains no source text, so turn that
    # invalid mutation into the only useful recovery action: read the current
    # file before attempting another repair.
    if tool.strip() == "write_file" and "content" not in arguments and "content_record" in arguments:
        path = arguments.get("path")
        if isinstance(path, str) and path:
            return Action(
                tool="read_file",
                arguments={"path": path, "start_line": 1, "end_line": 500},
            )
    if tool.strip() == "replace_text" and (
        ("old" not in arguments and "old_record" in arguments)
        or ("new" not in arguments and "new_record" in arguments)
    ):
        path = arguments.get("path")
        if isinstance(path, str) and path:
            return Action(
                tool="read_file",
                arguments={"path": path, "start_line": 1, "end_line": 500},
            )
    if tool.strip() in {"replace_lines", "replace_function"} and "content" not in arguments and "content_record" in arguments:
        path = arguments.get("path")
        if isinstance(path, str) and path:
            return Action(
                tool="read_file",
                arguments={
                    "path": path,
                    "start_line": int(arguments.get("start_line", 1)),
                    "end_line": int(arguments.get("end_line", 500)),
                },
            )
    return Action(tool=tool.strip(), arguments=arguments)


def compact_action(action: Action) -> str:
    """Keep source blobs out of the next prompt without creating fake source.

    Earlier versions replaced ``content`` with ``<stored N characters>``. Small
    local models sometimes copied that marker back into a later ``write_file``
    call, destroying a repair turn. A typed digest records that the write was
    made while being deliberately unusable as file content.
    """
    args: dict[str, Any] = {}
    for key, value in action.arguments.items():
        if key == "content" and isinstance(value, str):
            args["content_record"] = {
                "omitted": True,
                "characters": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
                "instruction": "use read_file before a later repair; this is not source text",
            }
        elif key in {"old", "new"} and isinstance(value, str) and len(value) > 1200:
            args[f"{key}_record"] = {
                "omitted": True,
                "characters": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
            }
        else:
            args[key] = value
    return json.dumps({"tool": action.tool, "arguments": args}, ensure_ascii=False)
