"""Read public instructions to check initial context size; no model or checker calls."""
import argparse
import json
import tempfile
from pathlib import Path

from qfa_agent.context import ContextManager
from qfa_agent.memory import WorkingMemory
from qfa_agent.prompts import system_prompt, user_prompt
from qfa_agent.strategies import route_task
from qfa_agent.task import load_task, expected_output_files
from qfa_agent.workflow import WorkflowController
from qfa_agent.workspace import TaskWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for unit in sorted((args.official_repo / 'units').iterdir()):
        if not (unit / 'card.toml').is_file():
            continue
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'output').mkdir(); (root/'scratch').mkdir()
            workspace = TaskWorkspace(unit.resolve(), root/'output', root/'scratch')
            task = load_task(unit)
            routing = route_task(task)
            inventory = workspace.inventory('input')
            messages = [{'role':'system', 'content':system_prompt(task, inventory, workspace.input_profile(inventory), routing)},
                        {'role':'user', 'content':user_prompt(task)}]
            workflow = WorkflowController(routing.strategy)
            memory = WorkingMemory(workspace, task.instruction_sha256, routing.strategy.category)
            checkpoint = memory.checkpoint(workflow={**workflow.snapshot(), 'guidance':workflow.guidance()},
                required_files=expected_output_files(task.safe_instruction), solver_digest=None,
                budget={'remaining_seconds':900, 'remaining_model_calls':18})
            _, stats = ContextManager(workspace).build(messages, checkpoint)
            rows.append({'task_id':unit.name, 'estimated_input_tokens':stats['estimated_input_tokens']})
    report = {'scope':'initial-context byte heuristic only; no model generation, pass@1, or guarantee for later trajectories',
              'n_tasks':len(rows), 'max_estimated_input_tokens':max(r['estimated_input_tokens'] for r in rows),
              'budget':16000, 'rows':rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
