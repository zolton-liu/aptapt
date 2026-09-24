"""Three-layer participant checks, separate from the official hidden grader.

Cross-checks recompute through library-owned code, not a solver's pass flag.
Scope/conventions are explicit. These receipts are not a security boundary
against hostile Python or a proof that an incorrectly specified task is right.
"""
from __future__ import annotations

import hashlib
import math
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import require_aligned, require_finite, require_correlation


LAYERS = ('structure', 'computation', 'cross_check')
_ISSUER = object()


class VerificationInputError(ValueError):
    """Malformed verification inputs, not evidence that a method is unsuitable."""


@dataclass(frozen=True)
class Check:
    layer: str
    name: str
    passed: bool
    evidence: str
    feedback: str = 'implementation'
    scope: str = ''
    _issued_by: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._issued_by is not _ISSUER:
            raise ValueError('use executable verification functions, not hand-written Check pass flags')


class VerificationFailure(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('verification failed: ' + ', '.join(c.name for c in report.checks if not c.passed))


class AuditReport:
    def __init__(self, checks):
        self.checks = tuple(checks)
        if (not self.checks or len(self.checks) > 16
                or not all(type(c) is Check and c.layer in LAYERS and type(c.passed) is bool
                           and c.feedback in {'implementation', 'method_review', 'undetermined'}
                           and isinstance(c.name, str) and 0 < len(c.name) <= 120
                           and isinstance(c.evidence, str) and 0 < len(c.evidence) <= 1000
                           and isinstance(c.scope, str) and len(c.scope) <= 500 for c in self.checks)):
            raise ValueError('audit requires bounded typed checks with executable evidence, not bool claims')
        if len({c.name for c in self.checks}) != len(self.checks):
            raise ValueError('audit check names must be unique')
        if any(not any(c.layer == layer for c in self.checks) for layer in LAYERS):
            raise ValueError('audit must cover structure, computation and cross_check; missing is not passing')

    def as_dict(self):
        failed = [c for c in self.checks if not c.passed]
        route = ('implementation' if any(c.feedback == 'implementation' for c in failed)
                 else 'method_review' if any(c.feedback == 'method_review' for c in failed)
                 else 'undetermined' if failed else 'none')
        return {'schema': 'three-layer-v1', 'passed': not failed,
                'layers': {layer: all(c.passed for c in self.checks if c.layer == layer) for layer in LAYERS},
                'checks': [{k: v for k, v in asdict(c).items() if not k.startswith('_')}
                           for c in self.checks], 'feedback_route': route,
                'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope': 'participant checks under declared conventions; not official correctness or root-cause proof'}

    def require_pass(self):
        if not all(c.passed for c in self.checks):
            raise VerificationFailure(self)


def _run(layer, name, fn, *, feedback='implementation', scope=''):
    try:
        evidence = fn()
        return Check(layer, name, True, str(evidence)[:1000], feedback, scope, _ISSUER)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, AssertionError, ArithmeticError, OSError, np.linalg.LinAlgError) as exc:
        route = 'implementation' if isinstance(exc, VerificationInputError) else feedback
        return Check(layer, name, False, f'{type(exc).__name__}: {exc}'[:1000], route, scope, _ISSUER)


def reference_crosscheck(reported, reference_fn, inputs, *, description, atol, rtol=0):
    """Fallback for unsupported domains; reference independence requires review."""
    def check():
        _require(callable(reference_fn) and isinstance(description, str) and bool(description.strip()),
                 'separate executable reference function and explanation required')
        reference = reference_fn(inputs)
        error = _close(reported, reference, atol, rtol)
        return f'executed reference={reference_fn.__name__}; max_error={error}; rationale={description[:240]}'
    return _run('cross_check', 'custom_reference_comparison', check,
                scope='model-authored reference: numeric recomputation checked, independence/correctness of reference not established by framework')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _input_require(condition, message):
    if not condition:
        raise VerificationInputError(message)


def _close(actual, expected, atol, rtol):
    _input_require(math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0, 'invalid tolerance')
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _input_require(a.shape == b.shape and a.size > 0, f'shape mismatch/empty: {a.shape} vs {b.shape}')
    require_finite(a, label='reported'); require_finite(b, label='recomputed')
    error = float(np.max(np.abs(a - b)))
    _require(np.allclose(a, b, atol=atol, rtol=rtol), f'max absolute discrepancy={error:.12g}')
    return error


def structure(values: dict, fields: dict, *, shapes=None, name='structure'):
    def check():
        _require(isinstance(values, dict) and isinstance(fields, dict) and bool(fields), 'explicit nonempty field schema required')
        for key, expected in fields.items():
            _require(key in values, f'missing field {key}')
            value = values[key]
            _require(type(value) is expected or (expected is float and type(value) is int),
                     f'{key}: expected {expected.__name__}, got {type(value).__name__}')
        for key, expected in (shapes or {}).items():
            actual = np.shape(values[key])
            _require(len(actual) == len(expected) and all(e is None or e == a for a, e in zip(actual, expected)),
                     f'{key}: shape {actual}, required {expected}')
        return f'checked fields={list(fields)}, shapes={shapes or {}}'
    return _run('structure', name, check)


def artifact_structure(output_dir, specifications):
    def finite_json(value):
        # JSON exponent overflow (1e309) is not caught by parse_constant.
        if isinstance(value, float):
            _require(math.isfinite(value), 'non-finite JSON number')
        elif isinstance(value, dict):
            for child in value.values(): finite_json(child)
        elif isinstance(value, list):
            for child in value: finite_json(child)

    def check():
        root = Path(output_dir).resolve()
        _require(isinstance(specifications, dict) and bool(specifications), 'nonempty output schema required')
        for filename, spec in specifications.items():
            path = root / filename
            _require(not path.is_symlink() and path.resolve().is_relative_to(root), 'output path escapes or is symlink')
            _require(path.is_file() and 0 < path.stat().st_size <= 8_000_000, f'{filename}: missing, empty or exceeds deep-check cap')
            if spec.get('format') == 'json':
                value = json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                finite_json(value)
                result = structure(value, spec['fields'], shapes=spec.get('shapes'))
                _require(result.passed, f'{filename}: {result.evidence}')
            elif spec.get('format') == 'csv':
                import csv
                with path.open(newline='') as stream:
                    rows = list(csv.reader(stream))
                columns = spec['columns']
                _require(bool(columns) and rows[0] == list(columns), f'{filename}: column names/order mismatch')
                if spec.get('rows') is not None:
                    _require(len(rows) - 1 == spec['rows'], f'{filename}: row count mismatch')
                for row in rows[1:]:
                    _require(len(row) == len(columns), f'{filename}: ragged rows')
                    for cell, expected in zip(row, columns.values()):
                        _require(expected in {'string', 'number', 'integer'}, 'unsupported CSV field type')
                        if expected != 'string':
                            number = float(cell)
                            _require(math.isfinite(number) and (expected != 'integer' or number.is_integer()), f'{filename}: invalid numeric field')
            else:
                raise ValueError('supported deep schemas: json/csv; use a task-specific validator for other formats')
        return f'checked actual files/fields/types/shapes: {list(specifications)}'
    return _run('structure', 'artifact_schema', check)


def computation(*, arrays: dict, aligned=(), correlations=(), name='computation'):
    def check():
        _require(bool(arrays) or bool(aligned) or bool(correlations), 'no computation checks specified')
        for label, values in arrays.items():
            require_finite(values, label=label)
        if aligned:
            require_aligned(*aligned)
        for matrix in correlations:
            require_correlation(matrix)
        return f'finite={list(arrays)}, aligned={len(aligned)}, correlations={len(correlations)}'
    return _run('computation', name, check)


def regression_crosscheck(X, y, coefficients, predictions, residuals, *, atol=1e-8, rtol=1e-7):
    def check():
        _require(isinstance(X, pd.DataFrame) and all(isinstance(v, pd.Series) for v in (y, coefficients, predictions, residuals)),
                 'labelled DataFrame/Series required; explicit constant column for intercept')
        require_aligned(y, X, predictions, residuals)
        _require(X.columns.is_unique and coefficients.index.equals(X.columns), 'coefficient labels/order differ from design columns')
        for label, value in [('X', X), ('y', y), ('coef', coefficients)]:
            require_finite(value, label=label)
        # Scalar summation is separate from the solver's vectorized X @ beta.
        predicted = np.array([math.fsum(float(a) * float(b) for a, b in zip(row, coefficients))
                              for row in X.to_numpy()])
        residual = np.array([float(a) - float(b) for a, b in zip(y, predicted)])
        ep = _close(predictions, predicted, atol, rtol)
        er = _close(residuals, residual, atol, rtol)
        return f'recomputed from X/coefficient inputs: prediction_error={ep}, residual_error={er}'
    return _run('cross_check', 'regression_reconstruction', check,
                scope='checks prediction/residual consistency, not optimality of fitted coefficients or causal validity')


def risk_crosscheck(losses, reported_var, reported_es, *, alpha, quantile_rule, es_rule, atol=1e-8, rtol=1e-7):
    def check():
        _require(np.ndim(losses) == 1 and len(losses) > 0 and 0 < alpha < 1, 'nonempty 1-D losses and alpha in (0,1) required')
        require_finite(losses, label='loss samples')
        _require(quantile_rule in {'linear', 'inverted_cdf'}, 'quantile convention must be explicit')
        _require(es_rule in {'mean_at_or_above_var', 'fractional_tail'}, 'ES convention must be explicit')
        ordered = sorted(float(x) for x in losses)
        n = len(ordered)
        if quantile_rule == 'linear':
            rank = (n - 1) * alpha; lo = math.floor(rank); hi = math.ceil(rank)
            var = ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])
        else:
            var = ordered[math.ceil(n * alpha) - 1]
        if es_rule == 'mean_at_or_above_var':
            tail = [x for x in ordered if x >= var]
            es = math.fsum(tail) / len(tail)
        else:
            mass = n * (1 - alpha); whole = math.floor(mass); fraction = mass - whole
            reverse = list(reversed(ordered))
            es = (math.fsum(reverse[:whole]) + (fraction * reverse[whole] if fraction else 0)) / mass
        error = _close([reported_var, reported_es], [var, es], atol, rtol)
        return f'ordered-sample recomputation: n={n}, alpha={alpha}, rules={quantile_rule}/{es_rule}, error={error}'
    return _run('cross_check', 'risk_tail_reconstruction', check,
                scope='positive-loss convention, explicit empirical quantile/ES rules; no horizon rescaling inferred')


def pricing_crosscheck(coarse, medium, fine, *, grids, boundary_actual, boundary_reference,
                       reference_description, atol, rtol=0):
    def check():
        _input_require(len(grids) == 3 and all(type(g) is int and g > 0 for g in grids)
                 and grids[0] < grids[1] < grids[2], 'three strictly refined grid sizes required')
        _input_require(isinstance(reference_description, str) and bool(reference_description.strip()), 'boundary reference/convention required')
        arrays = [np.asarray(v, dtype=float) for v in (coarse, medium, fine)]
        _input_require(arrays[0].shape == arrays[1].shape == arrays[2].shape and arrays[0].size > 0, 'grid result shapes must match')
        for a in arrays: require_finite(a, label='grid result')
        _input_require(math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0, 'invalid tolerance')
        d1 = float(np.max(np.abs(arrays[0] - arrays[1])))
        d2 = float(np.max(np.abs(arrays[1] - arrays[2])))
        _require(d2 <= d1 + atol, f'grid differences grow: {d1} -> {d2}')
        _close(arrays[1], arrays[2], atol, rtol)
        boundary_error = _close(boundary_actual, boundary_reference, atol, rtol)
        return f'grid differences={d1},{d2}; boundary_error={boundary_error}; reference={reference_description[:180]}'
    return _run('cross_check', 'pricing_refinement_and_boundary', check, feedback='method_review',
                scope='empirical refinement/declared boundary evidence only; failure may still be implementation error, not proof of asymptotic convergence')


def ledger_crosscheck(trades, marks, claimed_cash, claimed_equity, *, initial_cash,
                      initial_positions, atol=1e-8, rtol=1e-7):
    def check():
        _require(isinstance(marks, pd.DataFrame) and isinstance(marks.index, pd.DatetimeIndex)
                 and len(marks) > 0 and marks.index.is_unique and marks.index.is_monotonic_increasing,
                 'unique chronological datetime marks required')
        require_aligned(marks, claimed_cash, claimed_equity)
        _require(marks.columns.is_unique, 'duplicate mark assets')
        require_finite(marks, label='marks')
        _require(math.isfinite(initial_cash), 'invalid initial cash')
        positions = dict(initial_positions)
        _require(set(positions) <= set(marks.columns) and all(math.isfinite(v) for v in positions.values()), 'invalid initial holdings')
        events = list(trades)
        _require(all(set(t) == {'time', 'asset', 'quantity', 'price', 'fee'} for t in events),
                 'trade schema: time, asset, signed quantity, price, nonnegative fee; no undeclared cashflows/actions')
        events.sort(key=lambda t: pd.Timestamp(t['time']))
        for t in events:
            _require(t['asset'] in marks.columns and marks.index[0] <= pd.Timestamp(t['time']) <= marks.index[-1], 'trade outside marks/assets')
            _require(all(math.isfinite(t[k]) for k in ('quantity', 'price', 'fee')) and t['price'] > 0 and t['fee'] >= 0,
                     'invalid fill price/quantity/fee')
        cash = float(initial_cash); i = 0; cash_path = []; equity_path = []
        for timestamp, row in marks.iterrows():
            while i < len(events) and pd.Timestamp(events[i]['time']) <= timestamp:
                t = events[i]; i += 1
                cash -= t['quantity'] * t['price'] + t['fee']
                positions[t['asset']] = positions.get(t['asset'], 0.0) + t['quantity']
            cash_path.append(cash)
            equity_path.append(cash + math.fsum(q * float(row[a]) for a, q in positions.items()))
        ec = _close(claimed_cash, cash_path, atol, rtol)
        ee = _close(claimed_equity, equity_path, atol, rtol)
        return f'independent fill-ledger replay: fills={len(events)}, cash_error={ec}, equity_error={ee}'
    return _run('cross_check', 'ledger_reconstruction', check,
                scope='spot assets, signed fills processed before same-time marks; excludes corporate actions, external cashflows, margin and derivatives settlement')
