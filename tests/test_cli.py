import json
import subprocess
import sys
from pathlib import Path

import pytest

from monte_carlo_price_simulator.cli import run
from monte_carlo_price_simulator.market_data import load_csv_prices

from .conftest import REPO_ROOT


def test_cli_writes_reports_and_charts(synthetic_csv: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "reports"
    summary = run([
        "--csv", str(synthetic_csv), "--days", "60", "--paths", "2000",
        "--target", "120", "--output-dir", str(out),
    ])

    printed = capsys.readouterr().out
    assert "Value-at-Risk" in printed and "Expected Shortfall" in printed
    assert "P(finish above $120.00)" in printed
    for name in ("TEST_summary.json", "TEST_bands.csv", "TEST_fan.png", "TEST_terminal.png"):
        assert (out / name).stat().st_size > 0

    payload = json.loads((out / "TEST_summary.json").read_text())
    assert payload["summary"]["var"] == pytest.approx(summary.var)
    assert payload["simulation"] == {"days": 60, "paths": 2000, "seed": 42}


def test_cli_is_deterministic(synthetic_csv: Path, tmp_path: Path) -> None:
    args = ["--csv", str(synthetic_csv), "--paths", "500", "--no-plots", "--output-dir", str(tmp_path)]

    assert run(args) == run(args)


def test_cli_overrides_drift_and_vol(synthetic_csv: Path, tmp_path: Path, capsys) -> None:
    run(["--csv", str(synthetic_csv), "--drift", "0", "--vol", "0.3", "--paths", "500",
         "--no-plots", "--json", "--output-dir", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert payload["model"]["annual_drift"] == pytest.approx(0)
    assert payload["model"]["annual_vol"] == pytest.approx(0.3)


def test_cli_requires_a_source() -> None:
    with pytest.raises(SystemExit):
        run([])


def test_bundled_sample_runs_as_a_console_module(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "monte_carlo_price_simulator", "--csv",
         str(REPO_ROOT / "data" / "SPY.csv"), "--paths", "1000", "--no-plots",
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )

    assert result.stdout.startswith("Monte Carlo price simulation: SPY")
    assert (tmp_path / "SPY_summary.json").exists()


def test_load_csv_applies_lookback() -> None:
    prices = load_csv_prices(REPO_ROOT / "data" / "SPY.csv", lookback_days=90)

    assert 55 <= len(prices) <= 66
    assert prices.index.is_monotonic_increasing


def test_load_csv_rejects_missing_close_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("Date,Open\n2024-01-02,1\n")

    with pytest.raises(ValueError, match="no close column"):
        load_csv_prices(path)
