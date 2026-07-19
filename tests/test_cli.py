"""CLI smoke test.

Notes
-----
Pattern for all future CLI tests: invoke through CliRunner, assert on exit
code and output text. Every new command gets at least an exit-code-zero
test here the day it ships; the math behind the command is tested in its
own domain package, not through the CLI.
"""

from typer.testing import CliRunner

from ballast import __version__
from ballast.cli.app import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
