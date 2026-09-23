"""Load an official Track-1 task directory without mutating it."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)


@dataclass(frozen=True)
class TaskSpec:
    root: Path
    instruction: str
    safe_instruction: str
    card: dict[str, Any]
    task_id: str
    category: str
    difficulty: str
    timeout_sec: float
    data_cutoff: str
    canaries: tuple[str, ...]
    instruction_sha256: str


def _redact_canary_lines(text: str) -> str:
    safe_lines: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if "canary" in lowered or "never appear in training corpora" in lowered:
            continue
        safe_lines.append(_UUID_RE.sub("[REDACTED-IDENTIFIER]", line))
    return "\n".join(safe_lines).strip()


def expected_output_files(instruction: str) -> tuple[str, ...]:
    """Extract explicitly named deliverables from the public instruction.

    Only structural declaration lines inside bounded output sections are
    scanned for bare names so input files cannot accidentally become required
    outputs. Absolute legacy output paths are recognised throughout.
    """

    suffixes = r"json|csv|tsv|parquet|html|txt|png|pdf|py|xlsx"
    filename = rf"[A-Za-z0-9_.-]+\.(?:{suffixes})"
    found = set(
        re.findall(
            rf"(?:/app/output/|/output/|output/)({filename})",
            instruction,
            flags=re.IGNORECASE,
        )
    )

    # A task can contain multiple output-related sections (for example,
    # ``Required Output Files`` followed by ``Output Schemas``).  Bound each
    # section at the next heading of the same or a higher level.  Scanning from
    # the first ``## Output`` to EOF used to pull later input/appendix names
    # such as ``params.json`` into the output contract.
    headings = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", instruction))
    for index, heading in enumerate(headings):
        title = heading.group(2).strip().lower()
        plain_title = re.sub(r"[`*_]", "", title)
        is_output_heading = bool(
            re.match(
                r"^(?:(?:step|file)\s+\d+[a-z]?\s*[:.]\s*)?"
                r"(?:(?:required|expected|final)\s+)?"
                r"(?:outputs?\b|deliverables?\b|save\s+(?:all\s+)?outputs?\b)",
                plain_title,
            )
            or re.match(
                r"^\d+[a-z]?\.\s*(?:outputs?\b|save\s+(?:all\s+)?outputs?\b)",
                plain_title,
            )
        )
        if not is_output_heading:
            continue

        level = len(heading.group(1))
        end = len(instruction)
        for following in headings[index + 1 :]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        output_section = instruction[heading.start() : end]
        # Extract names only from structural declaration lines.  Schema prose
        # can legitimately refer back to input files, so treating every
        # backticked filename in the whole section as a deliverable is unsafe.
        for line in output_section.splitlines():
            stripped = line.strip()
            if re.match(r"^#{1,6}\s+", stripped):
                found.update(re.findall(filename, stripped, flags=re.IGNORECASE))
                continue
            if stripped.startswith("|"):
                first_cell = stripped.strip("|").split("|", 1)[0]
                found.update(re.findall(filename, first_cell, flags=re.IGNORECASE))
                continue
            list_match = re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)(.*)$", stripped)
            if list_match:
                body = list_match.group(1)
                first_name = re.search(filename, body, flags=re.IGNORECASE)
                if first_name:
                    prefix = re.sub(r"[`*_]", "", body[: first_name.start()]).strip()
                    if re.fullmatch(
                        r"(?:(?:file\s+\d+\s*:\s*)?"
                        r"(?:(?:/app)?/output/)?)?",
                        prefix,
                        flags=re.IGNORECASE,
                    ):
                        found.update(re.findall(filename, body, flags=re.IGNORECASE))
                continue
            action_match = re.search(
                r"\b(?:output files?|deliverables?|save|write|create)\b",
                stripped,
                flags=re.IGNORECASE,
            )
            if action_match:
                for name_match in re.finditer(filename, stripped, flags=re.IGNORECASE):
                    if name_match.start() > action_match.start():
                        found.add(name_match.group(0))

    # Verifier-owned files are sometimes documented in the Deliverables
    # section specifically to tell participants not to create them.  They are
    # not part of the agent contract even when an explicit /output path is
    # shown (the public exemplar documents reward.json this way).
    verifier_owned = {"reward.json", "reward.txt", "pytest_report.json"}
    excluded: set[str] = set(verifier_owned)
    for line in instruction.splitlines():
        if re.search(
            r"\b(?:do\s+not\s+write|written\s+automatically|verifier\s+owns?)\b",
            line,
            flags=re.IGNORECASE,
        ):
            excluded.update(re.findall(filename, line, flags=re.IGNORECASE))

    return tuple(sorted(found - excluded))


def load_task(root: Path) -> TaskSpec:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"task directory is not a directory: {root}")
    instruction_path = root / "instruction.md"
    card_path = root / "card.toml"
    if not instruction_path.is_file():
        raise ValueError(f"missing required task file: {instruction_path}")
    if not card_path.is_file():
        raise ValueError(f"missing required task file: {card_path}")

    instruction = instruction_path.read_text(encoding="utf-8", errors="replace")
    with card_path.open("rb") as handle:
        card = tomllib.load(handle)
    if not isinstance(card, dict):
        raise ValueError("card.toml must decode to an object")

    task_table = card.get("task", {})
    agent_table = card.get("agent", {})
    metadata = card.get("metadata", {})
    provenance = card.get("provenance", {})
    contamination = card.get("contamination", {})
    task_id = str(task_table.get("id", root.name))
    timeout = float(agent_table.get("timeout_sec", 1800.0))
    if timeout <= 0:
        raise ValueError("[agent].timeout_sec must be positive")

    found = set(match.lower() for match in _UUID_RE.findall(instruction))
    card_canary = contamination.get("canary_guid")
    if isinstance(card_canary, str) and card_canary:
        found.add(card_canary.lower())

    return TaskSpec(
        root=root,
        instruction=instruction,
        safe_instruction=_redact_canary_lines(instruction),
        card=card,
        task_id=task_id,
        category=str(metadata.get("category", "")),
        difficulty=str(metadata.get("difficulty", "")),
        timeout_sec=timeout,
        data_cutoff=str(provenance.get("data_cutoff", "")),
        canaries=tuple(sorted(found)),
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
    )
