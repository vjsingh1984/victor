import pytest
from victor.tools.unified.parser import (
    detect_shell_operators,
    shell_operator_rejection,
    split_command,
)


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
    assert args == [
        "fs",
        "write",
        "app.py",
        "--content",
        'def foo():\n    return "bar"\n',
    ]


def test_split_triple_single_quotes():
    cmd = "fs write app.py --content '''print('hello')\nprint(\"world\")'''"
    args = split_command(cmd)
    assert args == [
        "fs",
        "write",
        "app.py",
        "--content",
        "print('hello')\nprint(\"world\")",
    ]


def test_split_nested_quotes():
    cmd = 'fs write test.py -c """print("double")\nprint(\'single\')"""'
    args = split_command(cmd)
    assert args == [
        "fs",
        "write",
        "test.py",
        "-c",
        "print(\"double\")\nprint('single')",
    ]


def test_split_multiple_triple_quotes():
    cmd = 'fs patch -f app.py --search """old_code""" --replace """new_code"""'
    args = split_command(cmd)
    assert args == [
        "fs",
        "patch",
        "-f",
        "app.py",
        "--search",
        "old_code",
        "--replace",
        "new_code",
    ]


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


# ---------------------------------------------------------------------------
# Shell-operator detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["cat", "x", "||", "echo", "y"], "||"),
        (["cat", "x", "&&", "echo", "y"], "&&"),
        (["ls", ";", "pwd"], ";"),
        (["grep", "foo", "|", "sort"], "|"),
        (["cat", "x", "2>/dev/null"], "2>/dev/null"),  # redirect glued to path
        (["cat", "x", ">", "out.txt"], ">"),
        (["cat", "x", ">>", "log.txt"], ">>"),
        (["cat", "x", "&", "bg"], "&"),
    ],
)
def test_detect_shell_operators_flags_operators(tokens, expected):
    assert detect_shell_operators(tokens) == expected


def test_detect_shell_operators_clean_command_is_none():
    assert detect_shell_operators(["cat", "main.py", "--offset", "10"]) is None


def test_detect_shell_operators_ignores_operators_inside_quoted_token():
    # A `|` or `>` that is content inside a single token is NOT flagged.
    assert detect_shell_operators(["python", "a | b"]) is None
    assert detect_shell_operators(["grep", "x>=1"]) is None
    assert detect_shell_operators(["write", "-c", "print('a;b')"]) is None


def test_shell_operator_rejection_message_is_actionable():
    msg = shell_operator_rejection("fs", "||")
    assert "SHELL OPERATOR NOT SUPPORTED" in msg
    assert "`fs`" in msg
    assert "`shell` tool" in msg
    assert "`||`" in msg
