import hashlib
import json

import pytest

from qfa_agent.function_edit import replace_function_source
from qfa_agent.workspace import TaskWorkspace, ToolRouter
from qfa_agent.types import Action
from qfa_agent.agent import CodingAgent
from qfa_agent.protocol import parse_action, compact_action
from qfa_agent.context import ContextManager


def sha(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def test_replaces_complete_decorated_function_and_preserves_neighbors():
    before = '# 中文\nimport math\n@decorator\ndef compute(inputs):\n    return 1\n\n# retain\ndef other():\n    return 2\n'
    content = '@new_decorator\ndef compute(inputs):\n    return inputs["value"]\n'
    after = replace_function_source(before, 'compute', content, sha(before))
    assert after == '# 中文\nimport math\n' + content + '\n# retain\ndef other():\n    return 2\n'


@pytest.mark.parametrize('source', [
    'def f(): return 1\ndef f(): return 2\n',
    'def f(): return 1\nf = lambda: 2\n',
    'def f(): return 1\nif True:\n    f = lambda: 2\n',
    'def f(): return 1\nfrom other import f\n',
    'def f(): return 1\nimport other as f\n',
    'def f(): return 1\nclass f: pass\n',
    'class X:\n    def f(self): return 1\n',
])
def test_ambiguous_or_non_top_level_target_rejected(source):
    with pytest.raises(ValueError):
        replace_function_source(source, 'f', 'def f(): return 3\n', sha(source))


@pytest.mark.parametrize('content', ['def g(): return 2\n', 'def f(): return 2\nx=3\n',
                                    'def f():\nreturn 2\n', 'f=lambda:2', ''])
def test_rejects_wrong_shape_or_syntax_without_inference(content):
    source = 'def f(): return 1\n'
    with pytest.raises(ValueError):
        replace_function_source(source, 'f', content, sha(source))


def test_crlf_neighbors_and_async_are_supported():
    source = '# header\r\nasync def f():\r\n    return 1\r\nTAIL=3\r\n'
    actual = replace_function_source(source, 'f', 'async def f():\n    return 2', sha(source))
    assert actual == '# header\r\nasync def f():\n    return 2\nTAIL=3\r\n'


@pytest.fixture
def router(tmp_path):
    for area in ('input', 'scratch', 'output'):
        (tmp_path / area).mkdir()
    workspace = TaskWorkspace(*(tmp_path / area for area in ('input', 'output', 'scratch')))
    return ToolRouter(workspace)


def test_router_hash_stale_rejection_noop_and_candidate_digest(router):
    path = 'scratch/solve.py'
    source = 'def f(): return 1\n'
    router.workspace.write_file(path, source)
    read = router.dispatch(Action('read_file', {'path': path}))
    assert read.data['source_sha256'] == sha(source)
    args = dict(path=path, name='f', content='def f(): return 2\n', expected_sha256=sha(source))
    action = Action('replace_function', args)
    assert CodingAgent._candidate_solver_digest(router.workspace, action) == sha('def f(): return 2\n')
    result = router.dispatch(action)
    assert result.ok and result.mutated
    stale = router.dispatch(action)
    assert not stale.ok and not stale.mutated
    assert 'version changed' in stale.summary
    args['expected_sha256'] = sha('def f(): return 2\n')
    noop = router.dispatch(Action('replace_function', args))
    assert noop.ok and not noop.mutated


def test_invalid_edit_and_input_edit_preserve_bytes(router):
    source = 'def f(): return 1\n'
    router.workspace.write_file('scratch/solve.py', source)
    args = dict(path='scratch/solve.py', name='f', content='def f():\n  return (', expected_sha256=sha(source))
    result = router.dispatch(Action('replace_function', args))
    assert not result.ok and not result.mutated
    assert router.workspace.resolve('scratch/solve.py')[1].read_text() == source
    args.update(path='input/solve.py', content='def f(): return 2\n')
    result = router.dispatch(Action('replace_function', args))
    assert not result.ok and not result.mutated


def test_compact_history_recovers_to_read_and_offload_keeps_version(router):
    source = 'def f():\n' + ''.join(f'    # line {i}\n' for i in range(600)) + '    return 1\n'
    router.workspace.write_file('scratch/solve.py', source)
    action = Action('replace_function', dict(path='scratch/solve.py', name='f',
                    content='def f(): return 2\n', expected_sha256=sha(source)))
    assert parse_action(compact_action(action)).tool == 'read_file'
    result = router.dispatch(Action('read_file', dict(path='scratch/solve.py', start_line=1,end_line=600)))
    preview = json.loads(ContextManager(router.workspace, observation_chars=1500).observation(result))
    assert preview['data_preview']['source_sha256'] == sha(source)
