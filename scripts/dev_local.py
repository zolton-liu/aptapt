#!/usr/bin/env python3
"""Explicit local-only development entry point with Git-frozen evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIRS = {'src', 'scripts', 'tests', 'configs', 'knowledge', 'examples'}
SNAPSHOT_FILES = {'README.md', 'pyproject.toml', 'Dockerfile', 'LICENSE', '.gitignore', '.dockerignore'}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def git_state(root: Path, *, require_clean: bool = False) -> dict:
    status = git(root, 'status', '--porcelain', '--untracked-files=all')
    if require_clean and status:
        raise ValueError('uncommitted changes: review and commit before evaluation; nothing was started')
    return {'commit': git(root, 'rev-parse', 'HEAD'),
            'branch': git(root, 'branch', '--show-current'), 'dirty': bool(status)}


def load_profile(path: Path) -> dict:
    with path.open('rb') as stream:
        profile = tomllib.load(stream)
    model, evaluation = profile['model'], profile['evaluation']
    url = urllib.parse.urlsplit(model['endpoint'])
    if (url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or url.username or url.password or url.path not in {'', '/'}
            or url.query or url.fragment):
        raise ValueError('local profile must use a loopback HTTP origin, without credentials or paths')
    if not model['name'] or not re.fullmatch(r'[a-f0-9]{64}', model['digest']):
        raise ValueError('pin the installed local model name and full digest')
    for key in ('max_steps', 'max_response_tokens', 'model_timeout_sec', 'task_timeout_sec',
                'verifier_timeout_sec', 'context_max_tokens'):
        if type(evaluation[key]) is not int or evaluation[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if evaluation['max_response_tokens'] > 4000:
        raise ValueError('max_response_tokens cannot exceed the adapter cap of 4000')
    if evaluation['max_steps'] > 25:
        raise ValueError('max_steps cannot exceed the 25 model-call cap')
    for key in ('verification_required', 'stage_pipeline', 'auto_complete_valid_run',
                'experience_enabled'):
        if type(evaluation[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    return profile


def probe_model(profile: dict) -> dict:
    model = profile['model']
    # Explicitly bypass inherited proxies for a loopback-only development probe.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(model['endpoint'].rstrip('/') + '/api/tags', timeout=5) as response:
        installed = json.loads(response.read(2 * 1024 * 1024))['models']
    matches = [m for m in installed if m.get('name') == model['name']]
    if len(matches) != 1 or matches[0].get('digest') != model['digest']:
        raise ValueError('local model missing or digest changed; no fallback/download was attempted')
    return {'name': model['name'], 'digest': matches[0]['digest'], 'endpoint': model['endpoint']}


def local_environment(base: dict, profile: dict, source: Path) -> dict:
    # Prevent a forgotten replay, command adapter, or House config changing this experiment.
    env = {k: v for k, v in base.items()
           if not k.startswith(('QFA_', 'MODEL_', 'QFBENCH_'))}
    e, m = profile['evaluation'], profile['model']
    env.update(PYTHONPATH=str(source / 'src'), MODEL_ENDPOINT=m['endpoint'], MODEL_NAME=m['name'],
               MODEL_TOKEN='local-development-token', QFBENCH_SEED='0',
               NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1',
               QFA_DISABLE_STARTERS='1', QFA_JSON_MODE='1', QFA_CONTEXT_MEMORY='1',
               QFA_CONTEXT_MAX_TOKENS=str(e['context_max_tokens']),
               QFA_STAGE_PIPELINE=str(int(e['stage_pipeline'])),
               QFA_REPAIR_CONTEXT='1', QFA_VERSIONED_MEMORY='1',
               QFA_MAX_MODEL_CALLS=str(e['max_steps']), QFA_MAX_STEPS=str(e['max_steps']),
               QFA_MAX_RESPONSE_TOKENS=str(e['max_response_tokens']),
               QFA_VERIFICATION_REQUIRED=str(int(e['verification_required'])),
               QFA_AUTO_COMPLETE_VALID_RUN=str(int(e['auto_complete_valid_run'])))
    if e['experience_enabled']:
        env['QFA_EXPERIENCE_DIR'] = str(source / 'knowledge/verified-experiences')
    return env


def freeze_commit(root: Path, destination: Path, commit: str) -> dict:
    """Export selected tracked files from the commit, not the changing worktree."""
    archive = subprocess.check_output(['git', '-C', str(root), 'archive', '--format=tar', commit])
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as source:
        for member in source.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe Git archive path')
            if relative.parts[0] not in SNAPSHOT_DIRS and member.name not in SNAPSHOT_FILES:
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('source snapshots do not accept symlinks or special files')
            data = source.extractfile(member).read()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest[member.name] = hashlib.sha256(data).hexdigest()
    for required in ('scripts/evaluate_public_local.py', 'src/qfa_agent/cli.py'):
        if required not in manifest:
            raise ValueError(f'missing required source: {required}')
    return manifest


def smoke(profile: dict, model: dict, state: dict) -> dict:
    sys.path.insert(0, str(ROOT / 'src'))
    from qfa_agent.model import HouseEndpointModel, ModelError
    env = local_environment(dict(os.environ), profile, ROOT)
    env['QFA_MAX_RESPONSE_TOKENS'] = '64'
    previous = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        reply = HouseEndpointModel(model['endpoint'], model['name'], 0).complete(
            [{'role': 'user', 'content': 'Return exactly this JSON object: {"ok": true}'}],
            timeout_sec=90)
    except ModelError as exc:
        raise ValueError(f'local model smoke failed: {exc}') from exc
    finally:
        os.environ.clear()
        os.environ.update(previous)
    if json.loads(reply.content) != {'ok': True}:
        raise ValueError('model responded but failed the JSON connectivity smoke check')
    return {'scope': 'local model connectivity only; NOT a task evaluation or pass@1',
            'git': state, 'model': model, 'completed_responses': 1,
            'input_tokens': reply.input_tokens, 'output_tokens': reply.output_tokens,
            'passed': True}


def evaluate(args, profile: dict, model: dict, state: dict) -> int:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}', args.experiment):
        raise ValueError('experiment must be a new simple name, not a path')
    task_list = args.task_list.resolve()
    task_bytes = task_list.read_bytes()
    tasks = [line.strip() for line in task_bytes.decode().splitlines()
             if line.strip() and not line.strip().startswith('#')]
    if not tasks or len(set(tasks)) != len(tasks) or any(
            Path(t).name != t or t in {'.', '..'} for t in tasks):
        raise ValueError('task list must contain unique directory basenames')
    official = args.official_repo.resolve()
    if any(not (official / 'units' / t / 'card.toml').is_file() for t in tasks):
        raise ValueError('task list contains a missing public task')
    output_root = ROOT / 'eval_runs'
    if (output_root / args.experiment).exists():
        raise ValueError('experiment already exists; preserve it, do not overwrite or auto-resume')
    bundle = ROOT / 'work/local-runs' / args.experiment
    manifest = freeze_commit(ROOT, bundle / 'source', state['commit'])
    (bundle / 'task-list.txt').write_bytes(task_bytes)
    provenance = {'created_at': datetime.now(timezone.utc).isoformat(), 'git': state,
                  'model': model, 'profile': profile, 'python_version': sys.version,
                  'source_manifest': manifest, 'source_root': str(bundle / 'source'),
                  'task_list_sha256': hashlib.sha256(task_bytes).hexdigest(),
                  'scope': 'known public tasks, local model; not official House/hidden score'}
    provenance_path = bundle / 'provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    # Fail if a concurrent checkout/edit happened during preflight. The exported commit stays intact.
    if git_state(ROOT, require_clean=True) != state:
        raise ValueError('Git changed during preflight; preserved snapshot, no evaluation started')
    source = bundle / 'source'
    env = local_environment(dict(os.environ), profile, source)
    env['QFA_DEV_PROVENANCE_PATH'] = str(provenance_path)
    command = [sys.executable, str(source / 'scripts/evaluate_public_local.py'),
               '--official-repo', str(official), '--task-list', str(bundle / 'task-list.txt'),
               '--experiment', args.experiment, '--output-root', str(output_root),
               '--model-endpoint', model['endpoint'], '--model-name', model['name'], '--snapshot-inputs']
    for key in ('max_steps', 'max_response_tokens', 'model_timeout_sec', 'task_timeout_sec',
                'verifier_timeout_sec'):
        command += ['--' + key.replace('_', '-'), str(profile['evaluation'][key])]
    return subprocess.run(command, cwd=source, env=env, check=False).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/local-dev.toml')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='read-only Git/model checks; no generation')
    commands.add_parser('smoke', help='one small real local model response; not pass@1')
    run = commands.add_parser('eval', help='new fixed-list evaluation from a clean Git commit')
    run.add_argument('--experiment', required=True)
    run.add_argument('--task-list', type=Path, required=True)
    run.add_argument('--official-repo', type=Path, default=ROOT / 'work/track1-coding-public-full')
    args = parser.parse_args(argv)
    try:
        if os.environ.get('QFBENCH_NETWORK') == 'restricted':
            raise ValueError('local development launcher must not run in the official restricted runtime')
        state = git_state(ROOT, require_clean=args.command == 'eval')
        profile = load_profile(args.profile)
        model = probe_model(profile)
        if args.command == 'eval':
            return evaluate(args, profile, model, state)
        result = smoke(profile, model, state) if args.command == 'smoke' else {
            'git': state, 'model': model, 'ready_for_eval': not state['dirty']}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'local development stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
