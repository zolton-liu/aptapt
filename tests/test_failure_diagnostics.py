"""Failure routing must be invariant to source paths and unrelated warnings."""
import pytest

from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workflow import classify_failure, WorkflowController
from qfa_agent.strategies import STRATEGIES


@pytest.mark.parametrize('path', ['solve.py', 'finance_ops.py', 'nan.py',
                                  'schema/timeout/converge.py'])
@pytest.mark.parametrize('error,expected', [
    ("AttributeError: 'int' object has no attribute 'strip'", 'type'),
    ('ValueError: audit check names must be unique', 'type'),
    ('IndexError: index 5 is out of bounds for axis 1 with size 5', 'alignment'),
    ('MysteryException: unexplained failure', 'unknown'),
])
def test_source_path_does_not_change_exception_route(path, error, expected):
    output = f'Traceback (most recent call last):\n  File "{path}", line 39\n    finance_nan = schema_timeout\n{error}\n'
    assert classify_failure(ToolOutcome(False, 'Python program failed', {'output': output})) == expected


@pytest.mark.parametrize('warning', ['RuntimeWarning: overflow encountered in exp',
    'RuntimeWarning: invalid value encountered in arccos',
    '/tmp/finance_ops.py:39: RuntimeWarning: NaN encountered'])
def test_warning_does_not_override_terminal_implementation_failure(warning):
    output = warning + '\nTraceback\nValueError: audit check names must be unique\n'
    assert classify_failure(ToolOutcome(False, 'Python program failed', {'output': output})) == 'type'


def test_last_chained_exception_wins_and_structured_stage_is_preferred():
    data = {'output': 'ValueError: non-finite NaN\nDuring handling of the above exception:\n'
                      'Traceback\nAttributeError: diagnostic object has no attribute name'}
    assert classify_failure(ToolOutcome(False, 'failed', data)) == 'type'
    data['stage_evidence'] = {'stages': [{'name': 'compute', 'status': 'failed',
                                        'error': 'NameError: grid is not defined'}]}
    assert classify_failure(ToolOutcome(False, 'failed', data)) == 'symbol'


@pytest.mark.parametrize('error,expected', [
    ('ValueError: non-finite NaN in result', 'numeric'),
    ('ValueError: infinite value', 'numeric'),
    ('RuntimeError: solver did not converge', 'numeric'),
    ('LinAlgError: Singular matrix', 'numeric'),
    ('ValueError: Length of values (120) does not match length of index (8)', 'alignment'),
    ('AssertionError: balance does not match', 'invariant'),
    ('YFRateLimitError: Too Many Requests. Rate limited.', 'unknown'),
    ('ValueError: invalid finance configuration', 'type'),
    ('AttributeError: object has no attribute nan', 'type'),
])
def test_exception_variants(error, expected):
    assert classify_failure(ToolOutcome(False, 'Python program failed', {'output': error})) == expected


def test_numerical_exception_is_not_a_proven_method_failure():
    controller = WorkflowController(STRATEGIES['derivatives-pricing'])
    controller.observe(Action('run_python', {'script': 'scratch/solve.py'}),
                       ToolOutcome(False, 'failed', {'output': 'ValueError: non-finite NaN'}))
    assert controller.state.last_failure_kind == 'numeric'
    assert controller.feedback_route == 'undetermined'


def test_attribute_failure_routes_to_implementation_not_method_review():
    controller = WorkflowController(STRATEGIES['derivatives-pricing'])
    controller.observe(Action('run_python', {'script': 'scratch/solve.py'}),
                       ToolOutcome(False, 'failed', {'output': 'File finance_ops.py\nAttributeError: int has no attribute strip'}))
    assert controller.feedback_route == 'implementation'
    assert controller.guard(Action('revise_method', {})) is not None


def test_stdout_keywords_without_error_are_not_failure_proof():
    assert classify_failure(ToolOutcome(False, 'failed', {'output': 'finance_ops.py schema timeout NaN'})) == 'unknown'


def test_actual_timeout_flag_takes_precedence():
    assert classify_failure(ToolOutcome(False, 'failed', {'timed_out': True,
                           'output': 'ValueError: earlier warning'})) == 'timeout'
