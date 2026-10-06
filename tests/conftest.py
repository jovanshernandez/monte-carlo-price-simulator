from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """Two years of synthetic daily closes, written in the same shape as data/SPY.csv."""
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2024-01-02", periods=504)
    log_returns = 0.0004 + 0.012 * rng.standard_normal(len(dates))
    closes = 100 * np.exp(np.cumsum(log_returns))
    path = tmp_path / "test.csv"
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": closes.round(4)}).to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything tries to reach Yahoo Finance during tests."""

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr("monte_carlo_price_simulator.cli.fetch_close_prices", blocked)
