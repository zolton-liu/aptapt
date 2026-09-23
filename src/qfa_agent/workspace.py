"""Read-only task access, writable scratch/output access, and safe execution tools."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .executor import run_bounded
from .types import Action, OutputValidation, ToolOutcome


class WorkspaceError(ValueError):
    pass


_HIDDEN_INPUT_PARTS = {
    "checks",
    "reference",
    "reference_data",
    "solution",
    "solutions",
    "oracle",
    "dev",
    ".git",
    "__pycache__",
}
_HIDDEN_INPUT_FILES = {"manifest.json"}
_IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
_RESERVED_OUTPUTS = {"reward.json", "reward.txt", "pytest_report.json"}
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".html",
    ".xml",
    ".ini",
    ".cfg",
}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _finite_json(value: Any, location: str = "$") -> list[str]:
    issues: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(f"{location} contains non-finite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            issues.extend(_finite_json(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_finite_json(item, f"{location}[{index}]"))
    return issues


class TaskWorkspace:
    """Expose a deliberately small virtual filesystem to the model.

    Model paths always start with input/, scratch/, or output/.  Input scoring
    material is hidden even during local runs against the public repository.
    """

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        scratch_root: Path,
        *,
        canaries: Iterable[str] = (),
        max_read_bytes: int = 160_000,
    ):
        self.input_root = input_root.resolve(strict=True)
        self.output_root = output_root.resolve(strict=True)
        self.scratch_root = scratch_root.resolve(strict=True)
        self.canaries = tuple(value.lower().encode() for value in canaries if value)
        self.max_read_bytes = max_read_bytes
        if not self.input_root.is_dir():
            raise WorkspaceError("input root must be a directory")
        if _is_relative_to(self.output_root, self.input_root):
            raise WorkspaceError("output must not be inside the read-only task directory")
        if _is_relative_to(self.scratch_root, self.input_root):
            raise WorkspaceError("scratch must not be inside the read-only task directory")

    def _root_for_area(self, area: str) -> Path:
        roots = {
            "input": self.input_root,
            "scratch": self.scratch_root,
            "output": self.output_root,
        }
        try:
            return roots[area]
        except KeyError as exc:
            raise WorkspaceError("area must be input, scratch, or output") from exc

    def _normalise_model_path(self, raw: str) -> tuple[str, str]:
        if not isinstance(raw, str) or not raw.strip():
            raise WorkspaceError("path must be a non-empty string")
        raw = raw.strip().replace("\\", "/")
        if raw == "/input" or raw.startswith("/input/"):
            return "input", raw.removeprefix("/input").lstrip("/")
        if raw == "/output" or raw.startswith("/output/"):
            return "output", raw.removeprefix("/output").lstrip("/")
        if raw == "/app/output" or raw.startswith("/app/output/"):
            return "output", raw.removeprefix("/app/output").lstrip("/")
        if raw.startswith("/app/"):
            relative = raw.removeprefix("/app/")
            direct = self.input_root / relative
            nested = self.input_root / "environment" / "data" / relative.removeprefix("data/")
            if direct.exists():
                return "input", relative
            if nested.exists():
                return "input", nested.relative_to(self.input_root).as_posix()
            raise WorkspaceError(f"legacy path {raw!r} was not found under the task directory")
        if raw.startswith("/"):
            raise WorkspaceError("absolute paths are limited to /input, /output, and /app/output")
        first, separator, rest = raw.partition("/")
        if first not in {"input", "scratch", "output"}:
            raise WorkspaceError("model path must start with input/, scratch/, or output/")
        return first, rest if separator else ""

    def resolve(self, raw: str, *, write: bool = False, must_exist: bool = True) -> tuple[str, Path]:
        area, relative = self._normalise_model_path(raw)
        root = self._root_for_area(area)
        if write and area == "input":
            raise WorkspaceError("input is read-only")
        parts = Path(relative).parts
        lowered = {part.lower() for part in parts}
        if area == "input" and (
            lowered.intersection(_HIDDEN_INPUT_PARTS)
            or (parts and parts[-1].lower() in _HIDDEN_INPUT_FILES)
        ):
            raise WorkspaceError("verifier/reference material is not visible to the agent")
        if area == "output" and parts and parts[-1].lower() in _RESERVED_OUTPUTS:
            raise WorkspaceError("reward and pytest report files are reserved for the verifier")

        candidate = (root / relative).resolve(strict=False)
        if not _is_relative_to(candidate, root):
            raise WorkspaceError("path escapes its allowed root")
        if must_exist and not candidate.exists():
            raise WorkspaceError(f"path does not exist: {raw}")
        if write:
            parent = candidate.parent.resolve(strict=False)
            if not _is_relative_to(parent, root):
                raise WorkspaceError("parent directory escapes its allowed root")
        return area, candidate

    def inventory(self, area: str = "input", relative: str = "") -> tuple[str, ...]:
        root = self._root_for_area(area)
        virtual = f"{area}/{relative}" if relative else area
        _, start = self.resolve(virtual, must_exist=True)
        if not start.is_dir():
            return (start.relative_to(root).as_posix(),)
        files: list[str] = []
        for current, dirnames, filenames in os.walk(start, followlinks=False):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name.lower() not in _IGNORED_PARTS
                and not (area == "input" and name.lower() in _HIDDEN_INPUT_PARTS)
                and not (Path(current) / name).is_symlink()
            )
            for name in sorted(filenames):
                path = Path(current) / name
                if path.is_symlink():
                    continue
                rel = path.relative_to(root)
                if area == "input" and (
                    {part.lower() for part in rel.parts}.intersection(_HIDDEN_INPUT_PARTS)
                    or rel.name.lower() in _HIDDEN_INPUT_FILES
                ):
                    continue
                files.append(rel.as_posix())
        return tuple(files)

    def input_profile(
        self,
        inventory: tuple[str, ...],
        *,
        max_files: int = 24,
        max_chars: int = 32_000,
    ) -> str:
        """Return a bounded, deterministic schema/sample profile of task data.

        This gives the first model call enough information to choose an algorithm
        without spending several of the 25 House requests rediscovering small CSV
        headers and JSON parameter files. Verifier/reference paths are absent from
        ``inventory`` and are rejected again by ``resolve``.
        """

        sections: list[str] = []
        used = 0
        candidates = [
            rel for rel in inventory
            if rel not in {"instruction.md", "card.toml", "manifest.json"}
            and not rel.lower().endswith("dockerfile")
        ]
        for rel in candidates[:max_files]:
            _, path = self.resolve(f"input/{rel}")
            if not path.is_file():
                continue
            prefix = f"[{rel}] bytes={path.stat().st_size}"
            suffix = path.suffix.lower()
            detail = ""
            try:
                if suffix in {".csv", ".tsv"} and path.stat().st_size <= 8 * 1024 * 1024:
                    delimiter = "\t" if suffix == ".tsv" else ","
                    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                        reader = csv.reader(handle, delimiter=delimiter)
                        header = next(reader, [])
                        samples = []
                        row_count = 0
                        missing = [0 for _ in header]
                        first_values: list[str] = []
                        invalid_bid_ask = 0
                        lowered_header = [value.strip().lower() for value in header]
                        bid_index = lowered_header.index("bid") if "bid" in lowered_header else -1
                        ask_index = lowered_header.index("ask") if "ask" in lowered_header else -1
                        for row in reader:
                            row_count += 1
                            if len(samples) < 2:
                                samples.append(row)
                            if row:
                                first_values.append(row[0].strip())
                            for index in range(len(header)):
                                if index >= len(row) or not row[index].strip():
                                    missing[index] += 1
                            if bid_index >= 0 and ask_index >= 0 and max(bid_index, ask_index) < len(row):
                                try:
                                    invalid_bid_ask += float(row[bid_index]) >= float(row[ask_index])
                                except ValueError:
                                    pass
                    quality: dict[str, Any] = {}
                    if any(value != value.strip() for value in header):
                        quality["header_whitespace"] = True
                    nonzero_missing = {
                        header[index].strip(): count for index, count in enumerate(missing) if count
                    }
                    if nonzero_missing:
                        quality["missing_cells"] = nonzero_missing
                    duplicates = len(first_values) - len(set(first_values))
                    if duplicates:
                        quality["duplicate_first_field_raw"] = duplicates
                    if invalid_bid_ask:
                        quality["bid_gte_ask_rows"] = invalid_bid_ask
                    detail = (
                        f" rows={row_count} columns={json.dumps(header, ensure_ascii=False)}"
                        f" sample={json.dumps(samples, ensure_ascii=False)}"
                    )
                    if quality:
                        detail += f" deterministic_quality_flags={json.dumps(quality, ensure_ascii=False)}"
                elif suffix == ".json" and path.stat().st_size <= 2 * 1024 * 1024:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        payload = {
                            key: value for key, value in payload.items()
                            if "canary" not in str(key).lower()
                        }
                    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    if len(rendered) > 10_000:
                        if isinstance(payload, dict):
                            rendered = json.dumps(
                                {key: type(value).__name__ for key, value in payload.items()},
                                ensure_ascii=False,
                            )
                            rendered = f"top_level_types={rendered} (large JSON; read selectively)"
                        else:
                            rendered = f"top_level={type(payload).__name__} length={len(payload)}"
                    detail = f" content={rendered}"
                elif suffix == ".parquet":
                    try:
                        import pyarrow.parquet as parquet  # type: ignore[import-not-found]

                        parquet_file = parquet.ParquetFile(path)
                        fields = [
                            [field.name, str(field.type)]
                            for field in parquet_file.schema_arrow
                        ]
                        samples: list[dict[str, Any]] = []
                        if parquet_file.num_row_groups:
                            samples = parquet_file.read_row_group(0).slice(0, 2).to_pylist()
                        detail = (
                            f" rows={parquet_file.metadata.num_rows}"
                            f" row_groups={parquet_file.num_row_groups}"
                            f" schema={json.dumps(fields, ensure_ascii=False)}"
                            f" sample={json.dumps(samples, ensure_ascii=False, default=str)}"
                        )
                    except ImportError:
                        detail = " binary_table=true profile_note='inspect with generated Python/pyarrow'"
                    except Exception as exc:  # optional reader must not abort the agent
                        detail = f" binary_table=true profile_error={type(exc).__name__}"
                elif suffix in {".xlsx", ".xlsm"}:
                    try:
                        import openpyxl  # type: ignore[import-not-found]

                        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
                        sheet_profiles: list[dict[str, Any]] = []
                        try:
                            for sheet in workbook.worksheets[:6]:
                                preview = [
                                    list(row)
                                    for row in sheet.iter_rows(
                                        min_row=1, max_row=3, values_only=True
                                    )
                                ]
                                sheet_profiles.append(
                                    {
                                        "sheet": sheet.title,
                                        "max_row": sheet.max_row,
                                        "max_column": sheet.max_column,
                                        "preview": preview,
                                    }
                                )
                        finally:
                            workbook.close()
                        detail = " sheets=" + json.dumps(
                            sheet_profiles, ensure_ascii=False, default=str
                        )
                    except ImportError:
                        detail = " binary_workbook=true profile_note='inspect with generated Python/openpyxl'"
                    except Exception as exc:  # optional reader must not abort the agent
                        detail = f" binary_workbook=true profile_error={type(exc).__name__}"
                elif suffix in {".txt", ".md"} and path.stat().st_size <= self.max_read_bytes:
                    preview_lines = []
                    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                        lowered = line.lower()
                        if "canary" in lowered or "never appear in training corpora" in lowered:
                            continue
                        preview_lines.append(line)
                        if len(preview_lines) >= 8:
                            break
                    preview = "\n".join(preview_lines)
                    detail = f" preview={preview[:2000]!r}"
            except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
                detail = f" profile_error={type(exc).__name__}"
            section = prefix + detail
            for canary in self.canaries:
                if canary:
                    section = re.sub(
                        re.escape(canary.decode(errors="ignore")),
                        "[REDACTED-IDENTIFIER]",
                        section,
                        flags=re.IGNORECASE,
                    )
            if used + len(section) + 1 > max_chars:
                sections.append("[profile truncated at character budget]")
                break
            sections.append(section)
            used += len(section) + 1
        if len(candidates) > max_files:
            sections.append(f"[{len(candidates) - max_files} additional data files omitted]")
        return "\n".join(sections) or "(no separate task data files)"

    def read_file(self, raw: str, start_line: int = 1, end_line: int = 240) -> str:
        area, path = self.resolve(raw)
        if not path.is_file():
            raise WorkspaceError("read_file requires a regular file")
        with path.open("rb") as handle:
            prefix = handle.read(8192)
        if area == "input" and path.suffix.lower() in {".parquet", ".xlsx", ".xlsm"}:
            relative = path.relative_to(self.input_root).as_posix()
            return (
                "Binary tabular file; returning a deterministic schema/sample profile "
                "instead of raw bytes:\n"
                + self.input_profile((relative,), max_files=1, max_chars=24_000)
            )
        if b"\x00" in prefix:
            raise WorkspaceError("binary file cannot be read as prompt text; inspect it with Python")
        start = max(1, int(start_line))
        end = max(start, min(int(end_line), start + 499))
        if area == "input" and path.suffix.lower() in {".csv", ".tsv"} and path.stat().st_size > 64 * 1024:
            # The deterministic profile already contains row count, schema,
            # samples and quality flags. Keep ad-hoc reads from injecting
            # hundreds of tabular rows into every later model request.
            end = min(end, start + 29)
        selected: list[str] = []
        selected_bytes = 0
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if line_number < start:
                    continue
                if line_number > end:
                    break
                line = raw_line.rstrip("\r\n")
                lowered = line.lower()
                if area == "input" and (
                    "canary" in lowered or "never appear in training corpora" in lowered
                ):
                    continue
                if area == "input":
                    for canary in self.canaries:
                        line = re.sub(
                            re.escape(canary.decode()),
                            "[REDACTED-IDENTIFIER]",
                            line,
                            flags=re.IGNORECASE,
                        )
                rendered = f"{line_number}: {line}"
                encoded_size = len(rendered.encode("utf-8")) + 1
                if selected_bytes + encoded_size > self.max_read_bytes:
                    selected.append(f"{line_number}: [READ SLICE TRUNCATED AT BYTE LIMIT]")
                    break
                selected.append(rendered)
                selected_bytes += encoded_size
        return "\n".join(selected)

    def search_files(self, area: str, query: str, relative: str = "") -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query:
            raise WorkspaceError("search query must be non-empty")
        if len(query) > 500:
            raise WorkspaceError("search query is too long")
        matches: list[dict[str, Any]] = []
        root = self._root_for_area(area)
        for rel in self.inventory(area, relative):
            path = root / rel
            if path.suffix.lower() not in _TEXT_SUFFIXES or path.stat().st_size > self.max_read_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                lowered = line.lower()
                if area == "input" and (
                    "canary" in lowered or "never appear in training corpora" in lowered
                ):
                    continue
                if query in line:
                    for canary in self.canaries:
                        line = re.sub(
                            re.escape(canary.decode()),
                            "[REDACTED-IDENTIFIER]",
                            line,
                            flags=re.IGNORECASE,
                        )
                    matches.append({"path": f"{area}/{rel}", "line": line_number, "text": line[:500]})
                    if len(matches) >= 80:
                        return matches
        return matches

    def write_file(self, raw: str, content: str, *, overwrite: bool = False) -> Path:
        if not isinstance(content, str):
            raise WorkspaceError("content must be a string")
        if len(content.encode()) > 2_000_000:
            raise WorkspaceError("direct write is limited to 2 MB; generate large artifacts with run_python")
        _, path = self.resolve(raw, write=True, must_exist=False)
        if path.exists() and not overwrite:
            raise WorkspaceError("file already exists; use replace_text or set overwrite=true deliberately")
        if path.exists() and not path.is_file():
            raise WorkspaceError("write target is not a regular file")
        _atomic_write(path, content.encode("utf-8"))
        return path

    def copy_text_file(
        self, source_raw: str, destination_raw: str, *, overwrite: bool = False
    ) -> Path:
        """Copy a visible input template into scratch without exposing hidden files."""

        source_area, source = self.resolve(source_raw)
        destination_area, destination = self.resolve(
            destination_raw, write=True, must_exist=False
        )
        if source_area != "input" or destination_area != "scratch":
            raise WorkspaceError("copy_file is limited to visible input -> scratch")
        if not source.is_file() or source.suffix.lower() not in _TEXT_SUFFIXES:
            raise WorkspaceError(
                "copy_file requires a visible text source; binary tables are already profiled "
                "and must be loaded by a generated Python solver"
            )
        if source.stat().st_size > 2_000_000:
            raise WorkspaceError("copy_file source exceeds 2 MB")
        raw = source.read_bytes()
        if b"\x00" in raw[:8192]:
            raise WorkspaceError("copy_file source is binary")
        if destination.exists() and not overwrite:
            raise WorkspaceError("destination already exists; set overwrite=true deliberately")
        text = raw.decode("utf-8")
        for canary in self.canaries:
            if canary:
                text = re.sub(
                    re.escape(canary.decode(errors="ignore")),
                    "[REDACTED-IDENTIFIER]",
                    text,
                    flags=re.IGNORECASE,
                )
        if destination.suffix.lower() == ".py":
            replacements = {
                'Path("/app/data")': 'Path(os.environ["TASK_DIR"]) / "environment/data"',
                "Path('/app/data')": "Path(os.environ['TASK_DIR']) / 'environment/data'",
                'Path("/app/output")': 'Path(os.environ["OUTPUT_DIR"])',
                "Path('/app/output')": "Path(os.environ['OUTPUT_DIR'])",
                'Path("/input/environment/data")': 'Path(os.environ["TASK_DIR"]) / "environment/data"',
                "Path('/input/environment/data')": "Path(os.environ['TASK_DIR']) / 'environment/data'",
                'Path("/output")': 'Path(os.environ["OUTPUT_DIR"])',
                "Path('/output')": "Path(os.environ['OUTPUT_DIR'])",
            }
            original = text
            for old, new in replacements.items():
                text = text.replace(old, new)
            if text != original and not re.search(r"(?m)^import os\b|^from os\b", text):
                marker = "from pathlib import Path"
                if marker in text:
                    text = text.replace(marker, f"import os\n{marker}", 1)
        _atomic_write(destination, text.encode("utf-8"))
        return destination

    def replace_text(self, raw: str, old: str, new: str) -> Path:
        if not isinstance(old, str) or not old:
            raise WorkspaceError("old text must be a non-empty string")
        if not isinstance(new, str):
            raise WorkspaceError("new text must be a string")
        _, path = self.resolve(raw, write=True)
        if not path.is_file():
            raise WorkspaceError("replace target is not a regular file")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0 and "\n" in old:
            # ``read_file`` deliberately prefixes source with line numbers.
            # Small models sometimes copy that displayed slice verbatim into
            # ``old``.  Accept the unambiguous de-numbered equivalent instead
            # of wasting another model turn on the same mechanical mismatch.
            de_numbered = "\n".join(
                re.sub(r"^\s*\d+:\s?", "", line) for line in old.splitlines()
            )
            if de_numbered and text.count(de_numbered) == 1:
                match_end = text.index(de_numbered) + len(de_numbered)
                if (
                    match_end < len(text)
                    and text[match_end] == "\n"
                    and new.endswith(("\n", "\r"))
                ):
                    de_numbered += "\n"
                old = de_numbered
                count = 1
        if count != 1:
            raise WorkspaceError(f"old text must occur exactly once; found {count}")
        _atomic_write(path, text.replace(old, new, 1).encode("utf-8"))
        return path

    def replace_lines(
        self, raw: str, start_line: int, end_line: int, content: str
    ) -> Path:
        """Replace an exact inclusive line range in a writable text file."""

        if not isinstance(content, str):
            raise WorkspaceError("content must be a string")
        _, path = self.resolve(raw, write=True)
        if not path.is_file():
            raise WorkspaceError("replace_lines target is not a regular file")
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        start = int(start_line)
        end = int(end_line)
        if start < 1 or end < start or end > len(lines):
            raise WorkspaceError(
                f"line range must satisfy 1 <= start <= end <= {len(lines)}"
            )
        replacement = content
        if replacement and not replacement.endswith(("\n", "\r")) and end < len(lines):
            replacement += "\n"
        updated = "".join(lines[: start - 1]) + replacement + "".join(lines[end:])
        _atomic_write(path, updated.encode("utf-8"))
        return path

    def run_python(self, raw_script: str, args: list[str], timeout_sec: float) -> ToolOutcome:
        area, script = self.resolve(raw_script)
        if area not in {"scratch", "output"}:
            raise WorkspaceError("run_python executes only agent-created scratch/output scripts")
        if script.suffix.lower() != ".py" or not script.is_file():
            raise WorkspaceError("run_python requires an existing .py file")
        if len(args) > 32 or not all(isinstance(arg, str) and len(arg) <= 4096 for arg in args):
            raise WorkspaceError("args must contain at most 32 bounded strings")
        # Small/local coding models frequently use the path examples from the
        # task text literally.  Keep execution isolated in scratch, but expose
        # the common task-relative spellings as symlinks so plain ``open()``
        # calls resolve to the same read-only input and writable output roots
        # as the structured workspace tools.
        runtime_aliases = {
            self.scratch_root / "input": self.input_root,
            self.scratch_root / "output": self.output_root,
            # A frequent generated spelling is ``scratch/foo`` even though
            # execution already starts in scratch.  This compatibility alias
            # makes that path resolve without changing the task's input/output
            # isolation.
            self.scratch_root / "scratch": self.scratch_root,
        }
        legacy_task_dir = self.scratch_root / "task_dir"
        legacy_task_dir.mkdir(exist_ok=True)
        runtime_aliases.update(
            {
                legacy_task_dir / "environment": self.input_root / "environment",
                legacy_task_dir / "output": self.output_root,
            }
        )
        for alias, target in runtime_aliases.items():
            if target.exists() and not alias.exists():
                try:
                    alias.symlink_to(target, target_is_directory=True)
                except OSError:
                    pass

        data_root = self.input_root / "environment" / "data"
        if data_root.is_dir():
            by_name: dict[str, list[Path]] = {}
            for candidate in data_root.rglob("*"):
                if candidate.is_file() and not candidate.is_symlink():
                    by_name.setdefault(candidate.name, []).append(candidate)
            for name, candidates in by_name.items():
                alias = self.scratch_root / name
                if len(candidates) == 1 and not alias.exists():
                    try:
                        alias.symlink_to(candidates[0])
                    except OSError:
                        pass
        timeout = max(0.1, min(float(timeout_sec), 600.0))
        result = run_bounded(
            [sys.executable, "-m", "qfa_agent.runtime", str(script), *args],
            cwd=self.scratch_root,
            timeout_sec=timeout,
            extra_env={
                "TASK_DIR": str(self.input_root),
                "OUTPUT_DIR": str(self.output_root),
                "PYTHONPATH": os.pathsep.join(
                    (str(_PACKAGE_ROOT), str(self.scratch_root), str(self.output_root))
                ),
            },
        )
        return ToolOutcome(
            ok=result.passed,
            summary=("Python program completed" if result.passed else "Python program failed"),
            data={
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "duration_sec": result.duration_sec,
                "output": result.output,
                "truncated_bytes": result.truncated_bytes,
            },
            mutated=True,
        )

    def repair_known_runtime_issue(self, raw_script: str, runtime_output: str) -> dict[str, Any] | None:
        """Apply a narrow, traceback-authorized repair to ephemeral scratch code.

        These rules are deliberately small and only fire when the runtime itself
        names the remedy. They save a House request without guessing at finance
        semantics, and every applied edit is returned for the trajectory feedback.
        """

        area, script = self.resolve(raw_script)
        if area != "scratch" or script.suffix.lower() != ".py" or not script.is_file():
            return None
        if "format='mixed'" not in runtime_output and 'format="mixed"' not in runtime_output:
            return None
        if "doesn't match format" not in runtime_output or "pd.to_datetime" not in script.read_text(encoding="utf-8"):
            return None
        source = script.read_text(encoding="utf-8")
        pattern = re.compile(r"pd\.to_datetime\(([^()\n]+)\)")
        candidates = [match for match in pattern.finditer(source) if "format=" not in match.group(0)]
        if len(candidates) != 1:
            return None
        match = candidates[0]
        replacement = f"pd.to_datetime({match.group(1)}, format='mixed')"
        repaired = source[: match.start()] + replacement + source[match.end() :]
        compile(repaired, str(script), "exec")
        _atomic_write(script, repaired.encode("utf-8"))
        return {
            "rule": "pandas-mixed-datetime",
            "path": raw_script,
            "replacement": replacement,
            "sha256": hashlib.sha256(repaired.encode("utf-8")).hexdigest(),
        }

    def run_pytest(self, raw_paths: list[str], timeout_sec: float) -> ToolOutcome:
        if not raw_paths:
            raise WorkspaceError("run_pytest requires one or more agent-created test paths")
        resolved: list[str] = []
        for raw in raw_paths:
            area, path = self.resolve(raw)
            if area not in {"scratch", "output"}:
                raise WorkspaceError("official input checks are hidden; run only self-authored tests")
            resolved.append(str(path))
        timeout = max(0.1, min(float(timeout_sec), 600.0))
        result = run_bounded(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "--maxfail=3", *resolved],
            cwd=self.scratch_root,
            timeout_sec=timeout,
            extra_env={
                "TASK_DIR": str(self.input_root),
                "OUTPUT_DIR": str(self.output_root),
                "PYTHONPATH": os.pathsep.join(
                    (str(_PACKAGE_ROOT), str(self.scratch_root), str(self.output_root))
                ),
            },
        )
        return ToolOutcome(
            ok=result.passed,
            summary=("self-authored tests passed" if result.passed else "self-authored tests failed"),
            data={
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "duration_sec": result.duration_sec,
                "output": result.output,
                "truncated_bytes": result.truncated_bytes,
            },
        )

    def validate_outputs(self, expected_files: Iterable[str] = ()) -> OutputValidation:
        issues: list[str] = []
        warnings: list[str] = []
        files: list[str] = []
        total_bytes = 0
        for current, dirnames, filenames in os.walk(self.output_root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if not (Path(current) / name).is_symlink())
            for name in sorted(filenames):
                path = Path(current) / name
                rel = path.relative_to(self.output_root).as_posix()
                if path.is_symlink():
                    issues.append(f"output/{rel}: symbolic links are not allowed")
                    continue
                if not path.is_file():
                    continue
                files.append(rel)
                size = path.stat().st_size
                total_bytes += size
                if name.lower() in _RESERVED_OUTPUTS:
                    issues.append(f"output/{rel}: reserved verifier artifact")
                if size == 0:
                    issues.append(f"output/{rel}: empty file")
                if size > 256 * 1024 * 1024:
                    issues.append(f"output/{rel}: single file exceeds 256 MB safety cap")
                raw = path.read_bytes() if size <= 8 * 1024 * 1024 else b""
                if not raw and size > 8 * 1024 * 1024:
                    warnings.append(f"output/{rel}: skipped deep format inspection above 8 MB")
                lowered = raw.lower()
                for canary in self.canaries:
                    if canary and canary in lowered:
                        issues.append(f"output/{rel}: copied a task contamination identifier")
                        break
                suffix = path.suffix.lower()
                try:
                    if suffix == ".json" and raw:
                        parsed = json.loads(
                            raw.decode("utf-8"),
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                        )
                        issues.extend(f"output/{rel}: {item}" for item in _finite_json(parsed))
                    elif suffix in {".csv", ".tsv"} and raw:
                        dialect = "excel-tab" if suffix == ".tsv" else "excel"
                        rows = csv.reader(raw.decode("utf-8-sig").splitlines(), dialect=dialect)
                        header = next(rows, None)
                        if not header or not any(cell.strip() for cell in header):
                            issues.append(f"output/{rel}: missing table header")
                        elif any(not cell.strip() for cell in header):
                            issues.append(f"output/{rel}: table header contains a blank column")
                        elif len({cell.strip() for cell in header}) != len(header):
                            issues.append(f"output/{rel}: table header contains duplicate columns")
                        else:
                            non_finite = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
                            for row_number, row in enumerate(rows, start=2):
                                if len(row) != len(header):
                                    issues.append(
                                        f"output/{rel}: row {row_number} has {len(row)} fields; expected {len(header)}"
                                    )
                                    break
                                if any(cell.strip().lower() in non_finite for cell in row):
                                    issues.append(
                                        f"output/{rel}: row {row_number} contains a non-finite token"
                                    )
                                    break
                    elif suffix == ".py" and raw:
                        compile(raw, str(path), "exec")
                    elif suffix == ".parquet" and raw:
                        if len(raw) < 12 or not (raw.startswith(b"PAR1") and raw.endswith(b"PAR1")):
                            issues.append(f"output/{rel}: invalid Parquet magic bytes")
                    elif suffix == ".png" and raw and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                        issues.append(f"output/{rel}: invalid PNG signature")
                    elif suffix == ".pdf" and raw and not raw.startswith(b"%PDF-"):
                        issues.append(f"output/{rel}: invalid PDF signature")
                    elif suffix in {".xlsx", ".xlsm"} and raw and not raw.startswith(b"PK\x03\x04"):
                        issues.append(f"output/{rel}: invalid XLSX container signature")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, SyntaxError, csv.Error) as exc:
                    issues.append(f"output/{rel}: {type(exc).__name__}: {exc}")
        if not files:
            issues.append("no task deliverables were written")
        missing = sorted(set(expected_files) - set(files))
        if missing:
            issues.append("missing instruction-declared output files: " + ", ".join(missing))
        if len(files) > 1000:
            issues.append("more than 1000 output files were written")
        if total_bytes > 512 * 1024 * 1024:
            issues.append("total output exceeds 512 MB safety cap")
        if files and all(Path(name).suffix.lower() not in {".json", ".csv", ".tsv", ".parquet", ".py", ".html"} for name in files):
            warnings.append("output extensions are unusual; re-check the instruction")
        return OutputValidation(
            ok=not issues,
            files=tuple(sorted(files)),
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    def output_manifest(self) -> list[dict[str, Any]]:
        """Return bounded, deterministic artifact provenance for the risk gate."""

        manifest: list[dict[str, Any]] = []
        for rel in self.inventory("output")[:1000]:
            _, path = self.resolve(f"output/{rel}")
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
            item: dict[str, Any] = {
                "path": f"output/{rel}",
                "bytes": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            suffix = path.suffix.lower()
            try:
                if suffix == ".json" and size <= 8 * 1024 * 1024:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    item["shape"] = (
                        {"type": "object", "keys": sorted(value)[:80]}
                        if isinstance(value, dict)
                        else {"type": "array", "length": len(value)}
                        if isinstance(value, list)
                        else {"type": type(value).__name__}
                    )
                elif suffix in {".csv", ".tsv"} and size <= 8 * 1024 * 1024:
                    delimiter = "\t" if suffix == ".tsv" else ","
                    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                        reader = csv.reader(handle, delimiter=delimiter)
                        header = next(reader, [])
                        row_count = sum(1 for _ in reader)
                    item["shape"] = {"rows": row_count, "columns": header[:100]}
            except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
                item["profile_error"] = type(exc).__name__
            manifest.append(item)
        return manifest


class ToolRouter:
    def __init__(
        self,
        workspace: TaskWorkspace,
        *,
        max_python_runs: int = 8,
        max_pytest_runs: int = 3,
        expected_files: Iterable[str] = (),
    ):
        self.workspace = workspace
        self.max_python_runs = max_python_runs
        self.max_pytest_runs = max_pytest_runs
        self.python_runs = 0
        self.pytest_runs = 0
        self.expected_files = tuple(expected_files)

    @staticmethod
    def _str(args: dict[str, Any], key: str, default: str | None = None) -> str:
        value = args.get(key, default)
        if not isinstance(value, str):
            raise WorkspaceError(f"{key} must be a string")
        return value

    @staticmethod
    def _python_syntax_outcome(path: str, written: Path, verb: str) -> ToolOutcome | None:
        if written.suffix.lower() != ".py":
            return None
        source = written.read_text(encoding="utf-8")
        try:
            compile(source, path, "exec")
        except SyntaxError as exc:
            lines = source.splitlines()
            line_number = int(exc.lineno or 1)
            start = max(0, line_number - 3)
            end = min(len(lines), line_number + 2)
            context = "\n".join(
                f"{index + 1}: {lines[index]}" for index in range(start, end)
            )
            return ToolOutcome(
                False,
                f"{verb} {path}, but Python syntax validation failed",
                {
                    "error": exc.msg,
                    "line": line_number,
                    "offset": exc.offset,
                    "source_context": context,
                    "required_next_step": (
                        "Repair this syntax error before run_python; use the exact source_context "
                        "text for a unique replace_text action."
                    ),
                },
                mutated=True,
            )
        return None

    def dispatch(self, action: Action) -> ToolOutcome:
        args = action.arguments
        try:
            if action.tool == "list_files":
                area = self._str(args, "area", "input")
                relative = self._str(args, "path", "")
                files = self.workspace.inventory(area, relative)
                virtual_files = [f"{area}/{path}" for path in files[:500]]
                return ToolOutcome(True, f"found {len(files)} files", {"files": virtual_files})
            if action.tool == "read_file":
                path = self._str(args, "path")
                content = self.workspace.read_file(
                    path,
                    int(args.get("start_line", 1)),
                    int(args.get("end_line", 240)),
                )
                return ToolOutcome(True, f"read {path}", {"content": content})
            if action.tool == "search_files":
                area = self._str(args, "area", "input")
                query = self._str(args, "query")
                relative = self._str(args, "path", "")
                matches = self.workspace.search_files(area, query, relative)
                return ToolOutcome(True, f"found {len(matches)} matches", {"matches": matches})
            if action.tool == "write_file":
                path = self._str(args, "path")
                content = self._str(args, "content")
                if re.fullmatch(r"<stored \d+ characters>", content.strip()):
                    raise WorkspaceError(
                        "content is a compact-action placeholder, not file source; read the current "
                        "file and repair it with exact text instead"
                    )
                overwrite = args.get("overwrite", False)
                if not isinstance(overwrite, bool):
                    raise WorkspaceError("overwrite must be boolean")
                auto_overwrite = False
                # Small coding models often produce a complete repaired script but
                # omit the boolean flag. Scratch is ephemeral agent state, so a
                # substantial whole-Python-file rewrite is safe to accept there.
                # Output deliverables and small/ambiguous writes remain protected.
                if not overwrite and path.startswith("scratch/") and path.endswith(".py"):
                    _, candidate = self.workspace.resolve(path, write=True, must_exist=False)
                    if candidate.is_file():
                        overwrite = True
                        auto_overwrite = True
                written = self.workspace.write_file(path, content, overwrite=overwrite)
                syntax_outcome = self._python_syntax_outcome(path, written, "wrote")
                if syntax_outcome is not None:
                    return syntax_outcome
                return ToolOutcome(
                    True,
                    f"wrote {path}",
                    {
                        "bytes": written.stat().st_size,
                        "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
                        "auto_overwrite": auto_overwrite,
                    },
                    mutated=True,
                )
            if action.tool == "copy_file":
                source = self._str(args, "source")
                destination = self._str(args, "destination")
                overwrite = args.get("overwrite", False)
                if not isinstance(overwrite, bool):
                    raise WorkspaceError("overwrite must be boolean")
                written = self.workspace.copy_text_file(
                    source, destination, overwrite=overwrite
                )
                syntax_outcome = self._python_syntax_outcome(
                    destination, written, "copied"
                )
                if syntax_outcome is not None:
                    return syntax_outcome
                return ToolOutcome(
                    True,
                    f"copied {source} to {destination}",
                    {
                        "bytes": written.stat().st_size,
                        "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
                    },
                    mutated=True,
                )
            if action.tool == "replace_text":
                path = self._str(args, "path")
                old = self._str(args, "old")
                try:
                    written = self.workspace.replace_text(path, old, self._str(args, "new"))
                except WorkspaceError as exc:
                    if "old text must occur exactly once" not in str(exc):
                        raise
                    candidate_contexts: list[str] = []
                    try:
                        current = self.workspace.read_file(path, 1, 500)
                        lines = current.splitlines()
                        for index, line in enumerate(lines):
                            if old in line:
                                start = max(0, index - 1)
                                end = min(len(lines), index + 2)
                                candidate_contexts.append("\n".join(lines[start:end]))
                                if len(candidate_contexts) >= 8:
                                    break
                    except (WorkspaceError, OSError, UnicodeError):
                        pass
                    raw_source = ""
                    if not candidate_contexts:
                        try:
                            _, current_path = self.workspace.resolve(path)
                            raw_source = current_path.read_text(encoding="utf-8")[:12_000]
                        except (WorkspaceError, OSError, UnicodeError):
                            pass
                    return ToolOutcome(
                        False,
                        f"WorkspaceError: {exc}",
                        {
                            "candidate_contexts": candidate_contexts,
                            "current_source": raw_source,
                            "required_next_step": (
                                "Use replace_lines with exact line numbers from read_file, or expand old "
                                "to a unique multi-line context; do not repeat the same replace_text action."
                            ),
                        },
                    )
                syntax_outcome = self._python_syntax_outcome(path, written, "updated")
                if syntax_outcome is not None:
                    return syntax_outcome
                return ToolOutcome(
                    True,
                    f"updated {path}",
                    {"bytes": written.stat().st_size, "sha256": hashlib.sha256(written.read_bytes()).hexdigest()},
                    mutated=True,
                )
            if action.tool == "replace_lines":
                path = self._str(args, "path")
                content = self._str(args, "content")
                written = self.workspace.replace_lines(
                    path,
                    int(args.get("start_line", 0)),
                    int(args.get("end_line", 0)),
                    content,
                )
                syntax_outcome = self._python_syntax_outcome(path, written, "updated")
                if syntax_outcome is not None:
                    return syntax_outcome
                return ToolOutcome(
                    True,
                    f"updated lines in {path}",
                    {
                        "bytes": written.stat().st_size,
                        "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
                    },
                    mutated=True,
                )
            if action.tool == "run_python":
                script = self._str(args, "script")
                # Validate the target before charging the scarce execution
                # budget.  In particular, a hallucinated run before solve.py
                # exists must remain a recoverable build transition.
                area, resolved_script = self.workspace.resolve(script)
                if area not in {"scratch", "output"}:
                    raise WorkspaceError(
                        "run_python executes only agent-created scratch/output scripts"
                    )
                if resolved_script.suffix.lower() != ".py" or not resolved_script.is_file():
                    raise WorkspaceError("run_python requires an existing .py file")
                if self.python_runs >= self.max_python_runs:
                    raise WorkspaceError("run_python budget exhausted")
                raw_args = args.get("args", [])
                if not isinstance(raw_args, list):
                    raise WorkspaceError("args must be an array")
                self.python_runs += 1
                return self.workspace.run_python(
                    script,
                    raw_args,
                    float(args.get("timeout_sec", 120)),
                )
            if action.tool == "run_pytest":
                if self.pytest_runs >= self.max_pytest_runs:
                    raise WorkspaceError("run_pytest budget exhausted")
                paths = args.get("paths", [])
                if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
                    raise WorkspaceError("paths must be an array of strings")
                self.pytest_runs += 1
                return self.workspace.run_pytest(paths, float(args.get("timeout_sec", 120)))
            if action.tool == "validate_outputs":
                validation = self.workspace.validate_outputs(self.expected_files)
                return ToolOutcome(
                    validation.ok,
                    "output contract checks passed" if validation.ok else "output contract checks failed",
                    {
                        "files": list(validation.files),
                        "issues": list(validation.issues),
                        "warnings": list(validation.warnings),
                        "note": "Hidden task-specific pytest and financial invariants run only after the agent exits.",
                    },
                )
            return ToolOutcome(False, f"unknown tool: {action.tool}")
        except (WorkspaceError, OSError, UnicodeError, ValueError) as exc:
            return ToolOutcome(False, f"{type(exc).__name__}: {exc}")
