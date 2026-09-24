import pytest

from qfa_agent.verification import (AuditReport, AuditProtocolError, structure,
    computation, reference_crosscheck, VerificationFailure)
from qfa_agent.workflow import classify_failure
from qfa_agent.types import ToolOutcome


def test_reports_duplicate_missing_layer_and_failed_inputs_together():
    with pytest.raises(AuditProtocolError) as caught:
        AuditReport([structure([{'x': 1.0}], {'x': float}),
                     structure({'x': 1.0}, {'x': float}),
                     computation(arrays={'x': [1.0]})])
    error = str(caught.value)
    assert 'duplicates=' in error
    assert "missing=['cross_check']" in error
    assert 'explicit nonempty field schema required' in error
    assert classify_failure(ToolOutcome(False, 'failed', {'error': 'AuditProtocolError: ' + error})) == 'type'


def test_naming_fix_alone_does_not_accept_missing_crosscheck():
    with pytest.raises(AuditProtocolError, match="missing=\\['cross_check'\\]"):
        AuditReport([structure({'x': 1.0}, {'x': float}, name='summary'),
                     structure({'y': 2.0}, {'y': float}, name='results'),
                     computation(arrays={'x': [1.0]})])


def test_valid_assembly_still_executes_and_preserves_failed_checks():
    checks = [structure({'x': 2.0}, {'x': float}), computation(arrays={'x': [2.0]}),
              reference_crosscheck([2.0], lambda _: [1.0], {},
                                   description='test independent reference', atol=0)]
    report = AuditReport(checks)
    assert report.as_dict()['passed'] is False
    with pytest.raises(VerificationFailure):
        report.require_pass()


def test_bool_and_empty_claims_still_rejected():
    for checks in [[], [True], [{'passed': True}], [None]]:
        with pytest.raises(ValueError):
            AuditReport(checks)
