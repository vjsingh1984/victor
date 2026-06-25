import pytest
from victor.tools.unified.parser import split_command


def test_split_simple_command():
    args = split_command("fs ls /tmp")
    assert args == ["fs", "ls", "/tmp"]


def test_split_single_quotes():
    args = split_command("fs write app.py -c 'hello world'")
    assert args == ["fs", "write", "app.py", "-c", "hello world"]


def test_split_double_quotes():
    args = split_command('fs write app.py -c "hello world"')
    assert args == ["fs", "write", "app.py", "-c", "hello world"]


def test_split_triple_double_quotes():
    cmd = 'fs write app.py --content """def foo():\n    return "bar"\n"""'
    args = split_command(cmd)
    assert args == ["fs", "write", "app.py", "--content", 'def foo():\n    return "bar"\n']


def test_split_triple_single_quotes():
    cmd = "fs write app.py --content '''print('hello')\nprint(\"world\")'''"
    args = split_command(cmd)
    assert args == ["fs", "write", "app.py", "--content", "print('hello')\nprint(\"world\")"]


def test_split_nested_quotes():
    cmd = 'fs write test.py -c """print("double")\nprint(\'single\')"""'
    args = split_command(cmd)
    assert args == ["fs", "write", "test.py", "-c", "print(\"double\")\nprint('single')"]


def test_split_multiple_triple_quotes():
    cmd = 'fs patch -f app.py --search """old_code""" --replace """new_code"""'
    args = split_command(cmd)
    assert args == ["fs", "patch", "-f", "app.py", "--search", "old_code", "--replace", "new_code"]


def test_shlex_handles_escaped_quotes():
    cmd = 'fs write -c "escaped \\"quote\\" test"'
    args = split_command(cmd)
    assert args == ["fs", "write", "-c", 'escaped "quote" test']


def test_split_heredoc_as_single_argument():
    cmd = "code python <<'PY'\nprint('hello')\nprint(\"world\")\nPY"
    args = split_command(cmd)
    assert args == ["code", "python", "print('hello')\nprint(\"world\")"]


def test_split_heredoc_preserves_docstring_body():
    cmd = 'code python <<EOF\n"""module docstring"""\nprint("ok")\nEOF'
    args = split_command(cmd)
    assert args == ["code", "python", '"""module docstring"""\nprint("ok")']
