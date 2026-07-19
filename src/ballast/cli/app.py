"""CLI entry point.

Commands arrive per milestone (see _notes/IDEAS.md):
  v0.1.0  stats, import sentinel, ingest
  v0.2.0  cov compare
  v0.3.0  var, validate var
  v0.4.0  decompose
  v0.5.0  optimize, optimize compare
  v0.6.0  stress, rebalance, import sentinel-views

Notes
-----
This file stays small forever. Each domain gets its own module in cli/
(stats.py, cov.py, var.py, ...) defining its own Typer sub-app, registered
here with app.add_typer() -- the Sentinel pattern, one module per domain.

The root() callback is where global options land in v0.1.0 (--config,
--db). It also forces Typer into subcommand mode while only one command
exists: with a single command and no callback, Typer collapses the app and
`ballast version` breaks. Found by the scaffold's own test; don't remove.

CLI modules orchestrate and render; they contain no math. Anything worth
testing lives in the domain packages.
"""

import typer
from rich.console import Console

from ballast import __version__
from ballast.cli import data as data_cli
from ballast.cli import decompose as decompose_cli
from ballast.cli import rebalance as rebalance_cli
from ballast.cli import stats as stats_cli
from ballast.cli import stress as stress_cli
from ballast.cli import var as var_cli
from ballast.cli.cov import cov_app
from ballast.cli.optimize import optimize_app

app = typer.Typer(
    help="Ballast: portfolio construction and risk engine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def root() -> None:
    """Ballast: portfolio construction and risk engine."""
    # Global options (--config, --db) land here in v0.1.0.


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"ballast {__version__}")


# Command functions live in their domain modules (no math there either --
# see each module's notes); this file only registers them on the app.
app.command(name="stats")(stats_cli.stats)
app.command(name="ingest")(data_cli.ingest)
app.command(name="import-sentinel")(data_cli.import_sentinel)
app.add_typer(cov_app, name="cov")  # `ballast cov compare ...`
app.command(name="var")(var_cli.var)
app.add_typer(var_cli.validate_app, name="validate")  # `ballast validate var ...`
app.command(name="decompose")(decompose_cli.decompose)
app.add_typer(optimize_app, name="optimize")  # `ballast optimize compare ...`
app.command(name="stress")(stress_cli.stress)
app.command(name="rebalance")(rebalance_cli.rebalance)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
