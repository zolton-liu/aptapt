"""Executable boundary checks for generated solvers; never clean or guess silently."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


class ContractError(ValueError):
    pass


def _check_tolerance(atol: float) -> None:
    if not math.isfinite(atol) or atol < 0:
        raise ContractError('audit tolerance must be finite and non-negative')


def input_path(relative: str) -> Path:
    """Require an exact visible task-relative file, with no basename fallback."""
    from .workspace import TaskWorkspace, WorkspaceError
    workspace = TaskWorkspace(Path(os.environ['TASK_DIR']), Path(os.environ['OUTPUT_DIR']), Path.cwd())
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ContractError('input_path requires an exact task-relative path from the inventory')
    try:
        _, path = workspace.resolve('input/' + relative)
    except WorkspaceError as exc:
        raise ContractError(str(exc)) from exc
    if not path.is_file():
        raise ContractError(f'input file required: {relative}')
    return path


def json_object(relative: str, *, fields: dict[tuple[str, ...], type]) -> dict:
    """Validate explicit key paths/types, preserving scalar-versus-object distinctions."""
    value = json.loads(input_path(relative).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ContractError(f'{relative}: expected JSON object, got {type(value).__name__}')
    for key_path, expected in fields.items():
        if not isinstance(key_path, tuple) or not key_path or not all(isinstance(k, str) for k in key_path):
            raise ContractError('fields keys must be non-empty tuples of JSON keys')
        item = value
        for key in key_path:
            if not isinstance(item, dict) or key not in item:
                raise ContractError(f'{relative}: cannot access {key_path!r}; {key!r} missing or parent is {type(item).__name__}')
            item = item[key]
        valid = (type(item) in (int, float) if expected is float else type(item) is expected)
        if not valid:
            raise ContractError(f'{relative}: {key_path!r} expected {expected.__name__}, got {type(item).__name__}')
    return value


def require_columns(frame, columns) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing or not frame.columns.is_unique:
        raise ContractError(f'table columns invalid: missing={missing}, unique={frame.columns.is_unique}')


def require_aligned(*objects) -> None:
    """Check index identity before converting labelled data into arrays."""
    if len(objects) < 2:
        raise ContractError('alignment needs at least two labelled objects')
    first = objects[0]
    if not hasattr(first, 'index') or not first.index.is_unique:
        raise ContractError('alignment requires labelled objects with unique indexes')
    for other in objects[1:]:
        if (not hasattr(other, 'index') or not other.index.is_unique
                or not first.index.equals(other.index)):
            raise ContractError(f'index alignment mismatch: lengths {len(first)} and {len(other)}; align by task keys/dates explicitly, never truncate to min length')


def require_finite(value, *, label: str) -> None:
    import numpy as np
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f'{label}: expected numeric values') from exc
    if not np.isfinite(array).all():
        raise ContractError(f'{label}: non-finite numeric values; fix calculation/cleaning before writing outputs')


def require_correlation(matrix, *, atol: float = 1e-8) -> None:
    """Risk-model stage: symmetry, unit diagonal, bounds and PSD, without repair."""
    import numpy as np
    _check_tolerance(atol)
    require_finite(matrix, label='correlation')
    a = np.asarray(matrix, dtype=float)
    if (a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]
            or not np.allclose(a, a.T, atol=atol, rtol=0)
            or not np.allclose(np.diag(a), 1, atol=atol, rtol=0)
            or np.any(np.abs(a) > 1 + atol) or np.linalg.eigvalsh(a).min() < -atol):
        raise ContractError('correlation: require square/symmetric/PSD, unit diagonal and bounds [-1,1]')


def require_price_bounds(prices, lower, upper, *, atol: float = 1e-8) -> None:
    """Pricing stage: bounds must come from this contract, not guessed defaults."""
    import numpy as np
    _check_tolerance(atol)
    for label, value in [('prices', prices), ('lower bound', lower), ('upper bound', upper)]:
        require_finite(value, label=label)
    p, lo, hi = (np.asarray(x, dtype=float) for x in (prices, lower, upper))
    if any(x.ndim and x.shape != p.shape for x in (lo, hi)):
        raise ContractError('price bounds: non-scalar bounds must match price shape exactly')
    if np.any(lo > hi) or np.any(p < lo - atol) or np.any(p > hi + atol):
        raise ContractError('prices violate the explicitly supplied contract bounds')


def require_accounting(equity, cash, marked_holdings, *, atol: float = 1e-8) -> None:
    """Backtest stage: reconcile equally-shaped per-time ledger totals."""
    import numpy as np
    _check_tolerance(atol)
    for label, value in [('equity', equity), ('cash', cash), ('marked holdings', marked_holdings)]:
        require_finite(value, label=label)
    e, c, h = (np.asarray(x, dtype=float) for x in (equity, cash, marked_holdings))
    if e.shape != c.shape or e.shape != h.shape or not np.allclose(e, c + h, atol=atol, rtol=0):
        raise ContractError('ledger: shapes must match and equity must equal cash plus marked holdings')
