import numpy as np
import pandas as pd
import pytest

from monte_carlo_price_simulator.simulation import (
    GBMParams,
    SimulationConfig,
    estimate_parameters,
    simulate_paths,
)

PARAMS = GBMParams(s0=100.0, mu=0.0005, sigma=0.015)


def test_shape_and_starting_price() -> None:
    paths = simulate_paths(PARAMS, SimulationConfig(days=10, paths=3, seed=7))

    assert paths.shape == (11, 3)
    assert np.all(paths[0] == 100.0)
    assert np.all(paths > 0)


def test_same_seed_is_reproducible() -> None:
    config = SimulationConfig(days=30, paths=50, seed=42)

    np.testing.assert_array_equal(simulate_paths(PARAMS, config), simulate_paths(PARAMS, config))


def test_different_seeds_differ() -> None:
    a = simulate_paths(PARAMS, SimulationConfig(days=30, paths=50, seed=1))
    b = simulate_paths(PARAMS, SimulationConfig(days=30, paths=50, seed=2))

    assert not np.allclose(a, b)


def test_seeded_values_are_pinned() -> None:
    """Guards against accidental changes to how random numbers are consumed."""
    paths = simulate_paths(PARAMS, SimulationConfig(days=3, paths=2, seed=42))
    rng = np.random.default_rng(42)
    z = rng.standard_normal((3, 2))
    expected = 100.0 * np.exp(np.cumsum(PARAMS.log_drift + PARAMS.sigma * z, axis=0))

    np.testing.assert_allclose(paths[1:], expected)


def test_terminal_log_returns_match_theory() -> None:
    days, n = 252, 20_000
    paths = simulate_paths(PARAMS, SimulationConfig(days=days, paths=n, seed=2024))
    log_returns = np.log(paths[-1] / PARAMS.s0)

    expected_mean = (PARAMS.mu - 0.5 * PARAMS.sigma**2) * days
    expected_std = PARAMS.sigma * np.sqrt(days)
    standard_error = expected_std / np.sqrt(n)

    assert abs(log_returns.mean() - expected_mean) < 4 * standard_error
    assert log_returns.std(ddof=1) == pytest.approx(expected_std, rel=0.03)


def test_expected_terminal_price_is_s0_exp_mu_t() -> None:
    days = 60
    paths = simulate_paths(PARAMS, SimulationConfig(days=days, paths=50_000, seed=9))

    assert paths[-1].mean() == pytest.approx(PARAMS.s0 * np.exp(PARAMS.mu * days), rel=0.005)


def test_zero_volatility_is_deterministic_growth() -> None:
    params = GBMParams(s0=50.0, mu=0.001, sigma=0.0)
    paths = simulate_paths(params, SimulationConfig(days=5, paths=4, seed=None))

    np.testing.assert_allclose(paths[-1], 50.0 * np.exp(0.005))


def test_estimate_recovers_known_parameters() -> None:
    true = GBMParams(s0=100.0, mu=0.0006, sigma=0.02)
    series = simulate_paths(true, SimulationConfig(days=20_000, paths=1, seed=5))[:, 0]
    estimated = estimate_parameters(pd.Series(series))

    assert estimated.s0 == pytest.approx(series[-1])
    assert estimated.sigma == pytest.approx(true.sigma, rel=0.02)
    assert estimated.mu == pytest.approx(true.mu, abs=4 * true.sigma / np.sqrt(20_000))


def test_annualization_round_trip() -> None:
    params = GBMParams.from_annual(100.0, annual_mu=0.08, annual_sigma=0.2)

    assert params.annual_mu == pytest.approx(0.08)
    assert params.annual_sigma == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ([100, 100, 100], "non-zero volatility"),
        ([100, -1, 102], "positive"),
        ([100, 101], "at least three"),
    ],
)
def test_estimate_rejects_bad_input(prices: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_parameters(pd.Series(prices))


@pytest.mark.parametrize("config", [SimulationConfig(days=0), SimulationConfig(paths=0)])
def test_config_validation(config: SimulationConfig) -> None:
    with pytest.raises(ValueError):
        simulate_paths(PARAMS, config)
