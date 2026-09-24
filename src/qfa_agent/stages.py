"""A single-process staged solver protocol with explicit, version-bound evidence.

Audit checks are authored by the participant model, NOT official correctness
proofs. No intermediate-result cache survives a run or source edit.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path

from .verification import AuditReport, artifact_structure


STAGES = ('load_inputs', 'compute', 'audit', 'write_outputs')
_ACTIVE_STAGE = ContextVar('qfa_active_stage', default=None)


def callable_location(fn, script: Path) -> dict:
    """Locate the bound callable, not an earlier function with the same name."""
    code = getattr(fn, '__code__', None)
    if code is None or Path(code.co_filename).resolve() != script.resolve():
        return {'location_verified': False}
    lines = script.read_text(encoding='utf-8').splitlines()
    line = code.co_firstlineno
    try:
        relative = script.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        relative = script.name  # Direct unit tests may run outside runtime cwd.
    return {'location_verified': True, 'path': 'scratch/' + relative, 'line': line,
            'source_context': '\n'.join(f'{i+1}: {lines[i][:240]}'
                                        for i in range(max(0,line-2),min(len(lines),line+5))),
            'note': 'Actual bound callable; later definitions/assignments may replace earlier functions.'}


def staged_source(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False  # The normal Python execution reports the actual syntax error.
    return any(isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'QFA_STAGED' for t in node.targets)
               and isinstance(node.value, ast.Constant) and node.value.value is True
               for node in tree.body)


def run_stages(namespace: dict, script: Path, evidence_path: Path, run_id: str) -> None:
    verified_required = (namespace.get('QFA_VERIFICATION') == 'three-layer-v1'
                         or os.environ.get('QFA_VERIFICATION_REQUIRED', '0') == '1')
    state = {'mode': 'staged', 'run_id': run_id,
             'source_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
             'stages': [], 'complete': False,
             'audit_scope': 'self-authored executable checks; not the official verifier'}

    def save():
        evidence_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    inputs = result = None
    execution_id = object()
    for name in STAGES:
        entry = {'name': name, 'status': 'running'}
        state['stages'].append(entry)
        save()
        stage_token = _ACTIVE_STAGE.set((name, execution_id))
        try:
            fn = namespace.get(name)
            if not callable(fn):
                raise ValueError(f'staged solver requires function {name}')
            entry['callable_location'] = callable_location(fn, script)
            arguments = {'load_inputs': (), 'compute': (inputs,),
                         'audit': (inputs, result), 'write_outputs': (result,)}[name]
            expected = {'load_inputs': 'load_inputs()', 'compute': 'compute(inputs)',
                        'audit': 'audit(inputs, result)', 'write_outputs': 'write_outputs(result)'}[name]
            signature = None
            try:
                signature = inspect.signature(fn)
                signature.bind(*arguments)
            except (TypeError, ValueError) as exc:
                raise TypeError(f'stage protocol requires {expected}; got {name}{signature or " (unknown signature)"}. '
                                'Load paths/data in load_inputs and pass them in the inputs mapping; '
                                'do not add required parameters to the stage function.') from exc
            if name == 'load_inputs':
                inputs = fn()
                if not isinstance(inputs, Mapping) or not inputs:
                    raise ValueError('load_inputs must return a non-empty mapping of checked inputs')
            elif name == 'compute':
                result = fn(inputs)
                if not isinstance(result, Mapping) or not result:
                    raise ValueError('compute must return a non-empty mapping of results; do not write final outputs here')
            elif name == 'audit':
                checks = fn(inputs, result)
                if isinstance(checks, AuditReport):
                    entry['verification'] = checks.as_dict()
                    checks.require_pass()
                    entry['checks'] = {c.name: c.passed for c in checks.checks}
                elif verified_required:
                    raise ValueError('three-layer-v1 requires AuditReport, not self-declared bool checks')
                elif (not isinstance(checks, Mapping) or not checks or len(checks) > 64
                        or not all(isinstance(k, str) and 0 < len(k) <= 160 and type(v) is bool
                                   for k, v in checks.items())):
                    raise ValueError('audit must return 1..64 named, Python-bool checks derived from actual inputs/results')
                else:
                    entry['checks'] = dict(checks)
                    entry['verification'] = {'schema': 'legacy-booleans', 'independently_checked': False}
                    if not all(checks.values()):
                        raise ValueError('stage audit failed: ' + ', '.join(k for k, v in checks.items() if not v))
            else:
                fn(result)
                if verified_required:
                    structural = artifact_structure(os.environ['OUTPUT_DIR'], namespace.get('OUTPUT_SCHEMA'))
                    entry['artifact_check'] = {'passed': structural.passed, 'evidence': structural.evidence}
                    if not structural.passed:
                        raise ValueError('output structure failed: ' + structural.evidence)
            entry['status'] = 'passed'
        except Exception as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:1500])
            save()
            raise
        finally:
            _ACTIVE_STAGE.reset(stage_token)
        save()
    state['complete'] = True
    save()
