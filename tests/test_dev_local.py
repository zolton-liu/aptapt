import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dev_local', ROOT / 'scripts/dev_local.py')
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    for key, value in [('user.name', 'test'), ('user.email', 'test@example.invalid')]:
        subprocess.run(['git', '-C', str(root), 'config', key, value], check=True)
    for name, body in {'src/qfa_agent/cli.py': 'VERSION = 1\n',
                       'scripts/evaluate_public_local.py': '# runner\n',
                       '.gitignore': 'work/\neval_runs/\n.env\n',
                       'reports/private-local-note.md': 'not part of runtime snapshot\n'}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'fixture'], check=True)
    return root


def test_clean_git_gate_rejects_tracked_and_untracked_changes(repo):
    assert not dev.git_state(repo, require_clean=True)['dirty']
    (repo / 'src/qfa_agent/cli.py').write_text('VERSION = 2\n')
    with pytest.raises(ValueError, match='uncommitted'):
        dev.git_state(repo, require_clean=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'src'], check=True)
    with pytest.raises(ValueError, match='uncommitted'):
        dev.git_state(repo, require_clean=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'change'], check=True)
    (repo / 'untracked.py').write_text('x = 1\n')
    with pytest.raises(ValueError, match='uncommitted'):
        dev.git_state(repo, require_clean=True)


def test_freeze_reads_commit_not_later_edits_and_excludes_runtime_files(repo):
    state = dev.git_state(repo, require_clean=True)
    (repo / '.env').write_text('TEST_ONLY_TOKEN=not-a-real-credential\n')
    (repo / 'src/qfa_agent/cli.py').write_text('VERSION = 2\n')
    target = repo / 'work/frozen'
    manifest = dev.freeze_commit(repo, target, state['commit'])
    assert (target / 'src/qfa_agent/cli.py').read_text() == 'VERSION = 1\n'
    assert not (target / '.env').exists()
    assert not (target / 'reports').exists()
    assert 'src/qfa_agent/cli.py' in manifest
    with pytest.raises(FileExistsError):
        dev.freeze_commit(repo, target, state['commit'])


def test_snapshot_rejects_tracked_symlinks(repo):
    (repo / 'src/link.py').symlink_to('qfa_agent/cli.py')
    subprocess.run(['git', '-C', str(repo), 'add', 'src/link.py'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'link'], check=True)
    with pytest.raises(ValueError, match='symlinks'):
        dev.freeze_commit(repo, repo / 'work/frozen', dev.git_state(repo)['commit'])


def test_profile_is_local_only_and_pinned(tmp_path):
    profile = dev.load_profile(ROOT / 'configs/local-dev.toml')
    assert profile['model']['name'] == 'qwen2.5-coder:14b'
    raw = (ROOT / 'configs/local-dev.toml').read_text()
    for url in ['https://remote.invalid', 'http://127.0.0.1:11434/secret',
                'http://user:password@localhost:11434', 'http://localhost:11434?token=x']:
        p = tmp_path / 'profile.toml'
        p.write_text(raw.replace('http://127.0.0.1:11434', url))
        with pytest.raises(ValueError, match='loopback'):
            dev.load_profile(p)


def test_environment_clears_replay_remote_and_stale_feature_flags(tmp_path):
    profile = dev.load_profile(ROOT / 'configs/local-dev.toml')
    base = {'PATH': '/bin', 'QFA_REPLAY_FILE': 'old.json', 'QFA_MODEL_COMMAND': 'other',
            'MODEL_ENDPOINT': 'https://outside.invalid', 'MODEL_TOKEN': 'test-token',
            'QFA_VERIFICATION_REQUIRED': '0', 'QFBENCH_NETWORK': 'restricted'}
    env = dev.local_environment(base, profile, tmp_path)
    assert env['PATH'] == '/bin'
    assert env['MODEL_ENDPOINT'] == profile['model']['endpoint']
    assert env['MODEL_TOKEN'] == 'local-development-token'
    assert env['QFA_VERIFICATION_REQUIRED'] == '1'
    assert env['QFA_MAX_MODEL_CALLS'] == '18'
    assert not {'QFA_REPLAY_FILE', 'QFA_MODEL_COMMAND', 'QFBENCH_NETWORK'} & env.keys()
    assert env['QFA_EXPERIENCE_DIR'].startswith(str(tmp_path))


def test_evaluation_launches_frozen_runner_with_provenance(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(dev, 'ROOT', repo)
    unit = tmp_path / 'official/units/task-one'
    unit.mkdir(parents=True)
    (unit / 'card.toml').write_text('[task]\nid="task-one"\n')
    tasks = tmp_path / 'tasks.txt'
    tasks.write_text('task-one\n')
    args = SimpleNamespace(experiment='test-eval', task_list=tasks, official_repo=unit.parents[1])
    profile = dev.load_profile(ROOT / 'configs/local-dev.toml')
    state = dev.git_state(repo, require_clean=True)
    real_run = subprocess.run
    captured = {}

    def run(command, **kwargs):
        if command[0] == 'git':
            return real_run(command, **kwargs)
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev.subprocess, 'run', run)
    assert dev.evaluate(args, profile, profile['model'], state) == 0
    source = repo / 'work/local-runs/test-eval/source'
    assert captured['command'][1] == str(source / 'scripts/evaluate_public_local.py')
    assert captured['cwd'] == source
    assert captured['env']['PYTHONPATH'] == str(source / 'src')
    assert '--snapshot-inputs' in captured['command']
    provenance = json.loads(Path(captured['env']['QFA_DEV_PROVENANCE_PATH']).read_text())
    assert provenance['git']['commit'] == state['commit']
    assert provenance['profile'] == profile
    assert provenance['source_manifest']['src/qfa_agent/cli.py']
    with pytest.raises(FileExistsError):
        dev.evaluate(args, profile, profile['model'], state)


def test_restricted_runtime_cannot_start_local_model(monkeypatch):
    monkeypatch.setenv('QFBENCH_NETWORK', 'restricted')
    assert dev.main(['smoke']) == 2


def test_mismatched_model_digest_fails_without_fallback(monkeypatch):
    import io
    profile = dev.load_profile(ROOT / 'configs/local-dev.toml')
    class Opener:
        def open(self, *args, **kwargs):
            return io.BytesIO(json.dumps({'models': [{'name': profile['model']['name'],
                                                    'digest': '0' * 64}]}).encode())
    monkeypatch.setattr(dev.urllib.request, 'build_opener', lambda *args: Opener())
    with pytest.raises(ValueError, match='digest changed'):
        dev.probe_model(profile)
