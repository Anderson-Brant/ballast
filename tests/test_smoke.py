"""Every package imports; version is set.

Notes
-----
The import test is the canary: it catches syntax errors, circular imports,
and accidental heavy imports at module load (stub modules must stay
import-light so the CLI starts fast). PACKAGES must list every new module
as it's created -- a module missing from this list is invisible to CI
until something else happens to import it.
"""

import importlib

import ballast

PACKAGES = [
    "ballast.config",
    "ballast.cli.app",
    "ballast.data.prices",
    "ballast.data.sentinel_import",
    "ballast.portfolio.spec",
    "ballast.portfolio.stats",
    "ballast.cli.stats",
    "ballast.cli.data",
    "ballast.cli.var",
    "ballast.covariance.estimators",
    "ballast.covariance.harness",
    "ballast.data.edgar",
    "ballast.data.french",
    "ballast.data.sentinel_views",
    "ballast.factors.exposures",
    "ballast.factors.regression",
    "ballast.cli.cov",
    "ballast.risk.var",
    "ballast.risk.decompose",
    "ballast.cli.decompose",
    "ballast.validate.coverage",
    "ballast.optimize.mvo",
    "ballast.optimize.hrp",
    "ballast.optimize.risk_parity",
    "ballast.optimize.cvar",
    "ballast.optimize.black_litterman",
    "ballast.optimize.compare",
    "ballast.cli.optimize",
    "ballast.backtest.engine",
    "ballast.stress.scenarios",
    "ballast.portfolio.rebalance",
    "ballast.cli.stress",
    "ballast.cli.rebalance",
    "ballast.reporting.tables",
]


def test_version() -> None:
    assert ballast.__version__


def test_imports() -> None:
    for name in PACKAGES:
        importlib.import_module(name)
