"""Version-checked replacement of one unambiguous top-level Python function.

This edits model-supplied code only. It does not infer a repair or execute it.
"""
import ast
import hashlib
import re


def replace_function_source(source: str, name: str, content: str,
                            expected_sha256: str) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError('function name must be one top-level Python identifier')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('content must contain the complete replacement def, without line numbers')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if (not isinstance(expected_sha256, str)
            or not re.fullmatch(r'[a-f0-9]{64}', expected_sha256)):
        raise ValueError('expected_sha256 must be copied from the current read_file or repair_context')
    if expected_sha256 != digest:
        raise ValueError('source version changed; read the current file before replacing a function')
    try:
        tree = ast.parse(source)
        replacement = ast.parse(content)
    except SyntaxError as exc:
        raise ValueError(f'function edit requires valid Python source and replacement: {exc.msg}; '
                         'repair syntax with exact text first') from exc
    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    matches = [n for n in tree.body if isinstance(n, function_types) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f'expected one top-level def {name}; found {len(matches)}; no edit applied')
    target = matches[0]
    for statement in tree.body:
        if statement is target or isinstance(statement, function_types + (ast.ClassDef,)):
            # Other functions have their own local scope. A same-named class,
            # however, replaces the binding and makes function selection unsafe.
            if statement is not target and getattr(statement, 'name', None) == name:
                raise ValueError('function name has another top-level binding; use an exact text edit')
            continue
        if any((isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == name)
               or (isinstance(n, function_types + (ast.ClassDef,)) and n.name == name)
               or (isinstance(n, ast.alias) and (n.asname or n.name.split('.')[0]) == name)
               for n in ast.walk(statement)):
            raise ValueError('function name has another module binding; use an exact text edit')
    if (len(replacement.body) != 1 or not isinstance(replacement.body[0], function_types)
            or replacement.body[0].name != name):
        raise ValueError('replacement must be exactly one complete def with the same name')
    start = min([target.lineno, *(d.lineno for d in target.decorator_list)])
    end = target.end_lineno
    lines = source.splitlines(keepends=True)
    candidate = ''.join(lines[:start - 1]) + content.rstrip('\r\n') + '\n' + ''.join(lines[end:])
    compile(candidate, '<function-edit>', 'exec')
    return candidate
