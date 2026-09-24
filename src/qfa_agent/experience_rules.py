"""Reviewed procedural rules and executable reproductions; no task answers."""
from __future__ import annotations

import numpy as np
import pandas as pd


RULE = {
    'id': 'date-key-intersection-v1',
    'conditions': ['Task explicitly requires intersection of observation dates.',
                   'Both tables have unique datetime indexes.',
                   'Task permits joint complete-case filtering of these selected columns.'],
    'failure_evidence': 'Response/design rows differ after return calculation or missing-value removal.',
    'root_cause_hypothesis': 'The two sides retained different dates; dimensions alone do not prove this cause.',
    'repair': 'Join by date, apply a joint missingness mask, then require identical indexes before arrays.',
    'verification': 'Original shape-error reproduction plus missing-date, shuffled-date, duplicate-date and asymmetric-NaN cases.',
    'exclusions': ['As-of/publication-time joins', 'Entity/date panels without explicit multi-key adaptation',
                   'Tasks requiring union/imputation or a different missingness policy',
                   'Duplicate dates without a task-defined aggregation policy'],
    'tags': ['alignment'],
    'categories': ['factor-research', 'risk-management', 'cross-domain'],
    'fixture_scope': 'synthetic minimal reproduction of observed 2516/2515 shape error; not proof of the real Fama-French root cause',
}


def align_by_date(left, right, *, join_policy, missing_policy):
    if join_policy != 'intersection' or missing_policy != 'joint_complete_case':
        raise ValueError('unsupported policy; do not infer intersection/complete-case from a shape error')
    if not all(isinstance(x.index, pd.DatetimeIndex) and x.index.is_unique for x in (left, right)):
        raise ValueError('unique datetime indexes required; duplicate policy must be specified by task')
    dates = left.index.intersection(right.index).sort_values()
    a, b = left.loc[dates], right.loc[dates]
    def valid(x):
        missing = x.isna()
        return ~missing if x.ndim == 1 else ~missing.any(axis=1)
    keep = valid(a) & valid(b)
    a, b = a.loc[keep], b.loc[keep]
    if len(a) == 0 or not a.index.equals(b.index):
        raise ValueError('empty or inconsistent aligned sample')
    return a, b


def validate_rule():
    """Expected labels/values independently specified, not just no-exception tests."""
    dates = pd.date_range('2000-01-01', periods=2516)
    y = pd.Series(np.arange(2516, dtype=float), index=dates)
    x = pd.DataFrame({'x': np.arange(2515, dtype=float) + 1}, index=dates[1:])
    cases = []
    def record(name, fn):
        try:
            fn()
            cases.append({'case': name, 'passed': True})
        except Exception as exc:
            cases.append({'case': name, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'[:400]})
    def aligned(a, b):
        return align_by_date(a, b, join_policy='intersection', missing_policy='joint_complete_case')
    def original():
        try:
            y.to_numpy() - x.x.to_numpy()
        except ValueError:
            pass
        else:
            raise AssertionError('original dimension failure not reproduced')
        a, b = aligned(y, x)
        assert len(a) == 2515 and a.index.equals(dates[1:])
        np.testing.assert_array_equal(a.to_numpy(), b.x.to_numpy())
        assert not np.array_equal(y.to_numpy()[:2515], x.x.to_numpy())
    record('original_2516_2515', original)
    def missing_day():
        a, b = aligned(y.iloc[:5], x.iloc[:4].drop(dates[2]))
        assert a.index.equals(pd.DatetimeIndex([dates[1], dates[3], dates[4]]))
        np.testing.assert_array_equal(a, b.x)
    record('missing_day', missing_day)
    def shuffled():
        a, b = aligned(y.iloc[:5].iloc[::-1], x.iloc[:4].iloc[::-1])
        assert a.index.equals(dates[1:5]); np.testing.assert_array_equal(a, b.x)
    record('shuffled_dates', shuffled)
    def duplicates():
        duplicate = pd.concat([x.iloc[:3], x.iloc[:1]])
        try:
            aligned(y, duplicate)
        except ValueError as exc:
            assert 'unique datetime' in str(exc)
        else:
            raise AssertionError('duplicates were silently accepted')
    record('duplicate_dates_rejected', duplicates)
    def missingness():
        a0, b0 = y.iloc[:5].copy(), x.iloc[:4].copy()
        a0.loc[dates[1]] = np.nan; b0.loc[dates[3], 'x'] = np.nan
        a, b = aligned(a0, b0)
        assert a.index.equals(pd.DatetimeIndex([dates[2], dates[4]]))
        np.testing.assert_array_equal(a, b.x)
    record('asymmetric_missingness', missingness)
    return cases
