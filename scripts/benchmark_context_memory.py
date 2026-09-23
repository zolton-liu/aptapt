#!/usr/bin/env python3
"""Offline context stress benchmark; never a model pass@1 measurement."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from qfa_agent.agent import CodingAgent
from qfa_agent.context import ContextManager, estimate_tokens
from qfa_agent.memory import WorkingMemory
from qfa_agent.strategies import STRATEGIES, route_task
from qfa_agent.prompts import system_prompt, user_prompt
from qfa_agent.task import load_task, expected_output_files
from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workflow import WorkflowController
from qfa_agent.workspace import TaskWorkspace


def benchmark() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ('input', 'output', 'scratch'):
            (root / name).mkdir()
        workspace = TaskWorkspace(*(root / name for name in ('input', 'output', 'scratch')))
        workflow = WorkflowController(STRATEGIES['risk-management'])
        memory = WorkingMemory(workspace, 'synthetic-fixture', 'risk-management')
        context = ContextManager(workspace)
        initial = [{'role': 'system', 'content': 'Preserve schema and financial units.'},
                   {'role': 'user', 'content': 'Produce output/results.json with finite VaR.'}]
        old, new = list(initial), list(initial)
        for step in range(1, 4):
            action = Action('run_python', {'script': 'scratch/solve.py'})
            outcome = ToolOutcome(False, 'KeyError: close', {
                'output': ('price,date,volume\n' * 3000) + '\nKeyError: close',
                'source_context': '15: returns = df["close"].pct_change()',
            })
            memory.observe(step, action, outcome, 'source-v1')
            message = {'role': 'assistant', 'content': json.dumps({'tool': action.tool, 'arguments': action.arguments})}
            old.extend([message, {'role': 'user', 'content': CodingAgent._observation(outcome)}])
            new.extend([message, {'role': 'user', 'content': context.observation(outcome)}])
        old = CodingAgent._model_context(old, workflow=workflow, workspace=workspace,
                                        action_history=[], failure_memory=[])
        checkpoint = memory.checkpoint(workflow=workflow.snapshot(), required_files=('results.json',),
                                       solver_digest='source-v1', budget={'remaining_model_calls': 8})
        packed, stats = context.build(new, checkpoint)
        old_tokens = estimate_tokens(old)
        new_tokens = estimate_tokens(packed)
        return {
            'scope': 'synthetic context stress fixture; not pass@1 or provider-billed tokens',
            'baseline_estimated_input_tokens': old_tokens,
            'new_estimated_input_tokens': new_tokens,
            'estimated_reduction_percent': round(100 * (1 - new_tokens / old_tokens), 2),
            'deduplicated_active_failures': len(checkpoint['active_failures']),
            'repeated_failure_count': checkpoint['active_failures'][0]['count'],
            'instruction_preserved': packed[:2] == initial,
            **stats,
        }


def preflight(repo: Path) -> dict:
    """Build initial contexts from visible task inputs without running an LLM."""
    rows = []
    for card in sorted((repo / 'units').glob('*/card.toml')):
        task = load_task(card.parent)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'output').mkdir()
            (root / 'scratch').mkdir()
            workspace = TaskWorkspace(task.root, root / 'output', root / 'scratch', canaries=task.canaries)
            inventory = workspace.inventory('input')
            routing = route_task(task)
            messages = [{'role': 'system', 'content': system_prompt(
                task, inventory, workspace.input_profile(inventory), routing)},
                {'role': 'user', 'content': user_prompt(task)}]
            memory = WorkingMemory(workspace, task.instruction_sha256, routing.strategy.category)
            checkpoint = memory.checkpoint(
                workflow=WorkflowController(routing.strategy).snapshot(),
                required_files=expected_output_files(task.safe_instruction), solver_digest=None, budget={})
            try:
                _, stats = ContextManager(workspace).build(messages, checkpoint)
                rows.append({'task_id': task.task_id, 'fits': True,
                             'estimated_tokens': stats['estimated_input_tokens']})
            except ValueError as exc:
                rows.append({'task_id': task.task_id, 'fits': False, 'error': str(exc)})
    return {'n_tasks': len(rows), 'n_fit': sum(row['fits'] for row in rows),
            'max_estimated_tokens': max((row.get('estimated_tokens', 0) for row in rows), default=0),
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-repo', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {'context_stress': benchmark()}
    if args.official_repo:
        result['initial_context_preflight'] = preflight(args.official_repo)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    compact = {key: {k: v for k, v in value.items() if k != 'rows'} for key, value in result.items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
