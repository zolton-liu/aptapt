"""Adapt explicit-directory writers without treating their output as verified.

An immutable byte snapshot is readable during audit and publishable only in the
same staged run's write_outputs, after its AuditReport has passed. This is an
accidental-misuse guard, not a sandbox for hostile Python. The writer must honor
its directory argument; no output-path/env monkey-patching or scoring occurs.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .stages import _ACTIVE_STAGE


_ISSUER = object()
_MAX_BYTES = 64 * 1024 * 1024
_RESERVED = {'reward.json', 'reward.txt', 'pytest_report.json'}


def _stage(expected):
    active = _ACTIVE_STAGE.get()
    if active is None or active[0] != expected:
        raise ValueError(f'artifact operation requires the {expected} stage')
    return active[1]


@dataclass(frozen=True)
class ArtifactBundle:
    """Exact candidate bytes, never an assertion that calculations are correct."""
    files: tuple[tuple[str, bytes], ...]
    _run: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def __post_init__(self):
        if self._issuer is not _ISSUER:
            raise ValueError('use stage_writer to capture actual artifacts')

    def read_bytes(self, name: str) -> bytes:
        for path, data in self.files:
            if path == name:
                return data
        raise KeyError(name)

    def read_json(self, name: str):
        return json.loads(self.read_bytes(name))

    def read_csv(self, name: str, **kwargs):
        import pandas as pd
        return pd.read_csv(io.BytesIO(self.read_bytes(name)), **kwargs)


def stage_writer(writer) -> dict:
    """compute: stage_writer(lambda directory: legacy_writer(..., directory)).

    Return {'values': original return value, 'artifacts': ArtifactBundle}.
    Audit the actual bundle contents and independently recompute financial
    results. Capturing bytes does not satisfy any verification layer by itself.
    """
    execution_id = _stage('compute')
    if not callable(writer):
        raise TypeError('writer must accept one temporary directory argument')
    # Scratch-owned temporary files are removed after their bounded bytes are
    # captured. Repeated runs never reuse candidate artifacts from earlier runs.
    with tempfile.TemporaryDirectory(prefix='.qfa-candidate-', dir=Path.cwd()) as raw:
        directory = Path(raw)
        values = writer(directory)
        collected = []
        total = 0
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ValueError('candidate artifacts must not contain symlinks')
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError('candidate artifacts must be regular files')
            name = path.relative_to(directory).as_posix()
            if path.name.lower() in _RESERVED:
                raise ValueError('scoring files are not participant artifacts')
            size = path.stat().st_size
            if not size or total + size > _MAX_BYTES or len(collected) >= 256:
                raise ValueError('candidate artifact empty or snapshot budget exceeded')
            data = path.read_bytes()
            total += len(data)
            if total > _MAX_BYTES:
                raise ValueError('candidate snapshot budget exceeded')
            collected.append((name, data))
        if not collected:
            raise ValueError('writer produced no candidate artifacts')
    return {'values': values, 'artifacts': ArtifactBundle(tuple(collected), execution_id, _ISSUER)}


def publish_artifacts(bundle: ArtifactBundle) -> None:
    """write_outputs: publish the exact bytes inspected by audit, not a rerun."""
    execution_id = _stage('write_outputs')
    if (type(bundle) is not ArtifactBundle or bundle._issuer is not _ISSUER
            or bundle._run is not execution_id):
        raise ValueError('artifact bundle must come from this staged run')
    destination = Path(os.environ['OUTPUT_DIR']).resolve(strict=True)
    # Validate every target before writing any file; never traverse symlinks.
    targets = []
    for name, data in bundle.files:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.name.lower() in _RESERVED:
            raise ValueError('invalid candidate artifact path')
        target = destination / relative
        if any(p.is_symlink() for p in (target, *target.parents) if p != destination and destination in p.parents):
            raise ValueError('output symlinks are not allowed')
        if target.exists() and not target.is_file():
            raise ValueError('output target must be a regular file')
        targets.append((target, data))
    for target, data in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
