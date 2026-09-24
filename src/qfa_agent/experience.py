"""Offline experience promotion with actual regression evidence and scoped retrieval.

The solve loop may read a frozen, explicitly selected catalog. It never promotes
its own reflection, overwrites shared experience, or retrieves checker answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import experience_rules


RULE_ID = experience_rules.RULE['id']
REQUIRED_CASES = {'original_2516_2515', 'missing_day', 'shuffled_dates',
                  'duplicate_dates_rejected', 'asymmetric_missingness'}


def _binding():
    return hashlib.sha256(Path(experience_rules.__file__).read_bytes()).hexdigest()


def _engine_binding():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_digest():
    return hashlib.sha256(json.dumps(experience_rules.RULE, sort_keys=True).encode()).hexdigest()


class ExperienceStore:
    def __init__(self, root: Path):
        self.root = root

    @property
    def path(self):
        return self.root / f'{RULE_ID}.json'

    def _save(self, record):
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(self.path)

    def candidate(self, observation: str):
        if self.path.exists():
            raise ValueError('experience already exists; preserve provenance rather than overwrite it')
        if not isinstance(observation, str) or not 0 < len(observation) <= 2000:
            raise ValueError('bounded failure observation required')
        record = {'rule': experience_rules.RULE, 'status': 'candidate',
                  'observation': observation, 'observation_verified': False,
                  'actual_task_cause_verified': False, 'events': ['failure_recorded', 'root_cause_hypothesis'],
                  'created_at': datetime.now(timezone.utc).isoformat()}
        self._save(record)
        return record

    def verify(self):
        if sys.flags.optimize:
            raise ValueError('experience promotion requires assertions enabled; do not use Python -O')
        record = json.loads(self.path.read_text())
        if record['rule'] != experience_rules.RULE:
            raise ValueError('candidate semantics changed; this verifier cannot authorize another repair')
        cases = experience_rules.validate_rule()  # Never accept caller-supplied pass flags.
        original = next(c for c in cases if c['case'] == 'original_2516_2515')
        record['status'] = 'reproduction_verified' if original['passed'] else 'rejected'
        record['events'].append(record['status'])
        record['validation'] = {'cases': cases, 'rule_source_sha256': _binding(),
                                'promotion_engine_sha256': _engine_binding(),
                                'semantics_sha256': _canonical_digest(),
                                'scope': experience_rules.RULE['fixture_scope']}
        if (len(cases) == len(REQUIRED_CASES) and {c['case'] for c in cases} == REQUIRED_CASES
                and all(c['passed'] is True for c in cases)):
            record['events'].extend(['variant_regressions_passed', 'promoted_for_scoped_retrieval'])
            record['status'] = 'verified'
        else:
            record['status'] = 'rejected'
        record['verified_at'] = datetime.now(timezone.utc).isoformat()
        self._save(record)
        return record

    def retrieve(self, *, category: str, failure_kind: str):
        try:
            if not self.path.is_file() or self.path.stat().st_size > 32_000:
                return []
            raw = self.path.read_bytes()
            if len(raw) > 32_000:
                return []
            record = json.loads(raw)
            v = record.get('validation', {})
            cases = v.get('cases', [])
            if (record.get('status') != 'verified' or record.get('rule') != experience_rules.RULE
                    or v.get('rule_source_sha256') != _binding() or v.get('semantics_sha256') != _canonical_digest()
                    or v.get('promotion_engine_sha256') != _engine_binding()
                    or len(cases) != len(REQUIRED_CASES) or {c['case'] for c in cases} != REQUIRED_CASES
                    or not all(c['passed'] is True for c in cases)
                    or category not in experience_rules.RULE['categories']
                    or failure_kind not in experience_rules.RULE['tags']):
                return []
            # Do not inject raw observations or unverified root-cause claims.
            return [{**experience_rules.RULE, 'status': 'validated_on_listed_fixtures',
                     'applicability_to_current_task': 'must be checked; not automatically established',
                     'catalog_sha256': hashlib.sha256(raw).hexdigest()}]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['candidate', 'verify'])
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--observation')
    args = parser.parse_args()
    store = ExperienceStore(args.store)
    record = store.candidate(args.observation) if args.action == 'candidate' else store.verify()
    print(json.dumps({'rule': RULE_ID, 'status': record['status'], 'path': str(store.path)}))
    return 0 if record['status'] != 'rejected' else 1


if __name__ == '__main__':
    raise SystemExit(main())
