"""Shared fixtures.

Notes
-----
Here now:
- write_portfolio: tmp_path factory turning a dict into a portfolio.yaml
  and returning its path; spec tests build inputs through it.
- true_cov / synthetic_returns: a KNOWN daily covariance matrix and a long
  seeded sample drawn from it, so estimator tests compare output against
  ground truth instead of against another estimator.

Design rule: fixtures are deterministic -- seeded RNG, fixed dates. A test
that fails only sometimes is worse than no test.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def write_portfolio(tmp_path: Path) -> Callable[[dict], Path]:
    """Write a dict as portfolio.yaml in tmp_path; return the file path."""

    def _write(data: dict) -> Path:
        path = tmp_path / "portfolio.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


@pytest.fixture
def true_cov() -> pd.DataFrame:
    """A known 3-asset DAILY covariance: vols 2%/1.5%/1%, mild correlations."""
    vols = np.array([0.02, 0.015, 0.01])
    corr = np.array(
        [
            [1.0, 0.4, 0.0],
            [0.4, 1.0, 0.3],
            [0.0, 0.3, 1.0],
        ]
    )
    cov = np.outer(vols, vols) * corr
    symbols = ["AAA", "BBB", "CCC"]
    return pd.DataFrame(cov, index=symbols, columns=symbols)


@pytest.fixture
def synthetic_returns(true_cov: pd.DataFrame) -> pd.DataFrame:
    """8000 seeded draws from true_cov: enough data that every estimator
    should land close to the truth (the recovery test)."""
    rng = np.random.default_rng(42)  # fixed seed: same panel every run
    values = rng.multivariate_normal(np.zeros(3), true_cov.to_numpy(), size=8000)
    index = pd.bdate_range("2000-01-03", periods=8000)
    return pd.DataFrame(values, index=index, columns=true_cov.columns)
