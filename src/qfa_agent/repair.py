"""Bounded repair evidence from participant-owned traceback frames, not library lines."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


def repair_context(workspace, output: str) -> dict:
    frames = re.findall(r'File "([^"\n]+)", line (\d+)', output)
    owned = []
    for filename, line in frames:
        path = Path(filename)
        if not path.is_absolute():
            path = workspace.scratch_root / path
        path = path.resolve(strict=False)
        for area, root in [('scratch', workspace.scratch_root), ('output', workspace.output_root)]:
            try:
                virtual = f'{area}/{path.relative_to(root).as_posix()}'
                _, checked = workspace.resolve(virtual)
                if checked.suffix == '.py' and checked.stat().st_size <= workspace.max_read_bytes:
                    owned.append((virtual, checked, int(line)))
            except (ValueError, OSError):
                continue
    if not owned:
        return {'location_verified': False,
                'note': 'No participant-owned traceback frame; do not apply library line numbers to solve.py.'}
    virtual, path, line = owned[-1]
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines()
    locations = set(range(max(1, line - 3), min(len(lines), line + 3) + 1))
    primary_lines = sorted(locations)
    names: set[str] = set()
    try:
        tree = ast.parse(source)
        # Only expressions/statements on the faulting line, not the enclosing
        # function's entire subtree. Include definitions AND other call sites.
        for node in ast.walk(tree):
            if getattr(node, 'lineno', -1) == line and isinstance(node, ast.Name):
                names.add(node.id)
        related = sorted({node.lineno for node in ast.walk(tree)
                          if isinstance(node, ast.Name) and node.id in names})
        for number in related[:18]:
            locations.update(range(max(1, number - 1), min(len(lines), number + 1) + 1))
    except SyntaxError:
        pass
    ordered_lines = primary_lines + sorted(locations.difference(primary_lines))
    excerpt = '\n'.join(f'{n}: {lines[n - 1][:280]}' for n in ordered_lines)[:5000]
    packet = {
        'location_verified': True, 'path': virtual, 'line': line,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'expression': lines[line - 1][:500] if 1 <= line <= len(lines) else '',
        'related_symbols': sorted(names)[:12], 'source_context': excerpt,
        'library_tail': output[-1600:],
        'scope': 'Observed source references, not proof of runtime values or a computed patch.',
    }
    missing = re.search(r"No such file or directory: ['\"]([^'\"]+)['\"]", output)
    if missing:
        basename = Path(missing.group(1)).name
        packet['visible_path_candidates'] = [
            f'input/{rel}' for rel in workspace.inventory('input') if Path(rel).name == basename
        ][:8]
        packet['path_advice'] = 'Use the observed path; inspect all related assignments and call sites. No path was automatically substituted.'
    return packet
