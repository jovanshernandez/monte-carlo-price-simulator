import math

import numpy as np
import pytest

from monte_carlo_price_simulator.risk import (
    expected_shortfall,
    percentile_bands,
    probability_above,
    probability_below,
    summarize,
    value_at_risk,
)
from monte_carlo_price_simulator.simulation import GBMParams, SimulationConfig, simulate_paths

PARAMS = GBMParams.from_annual(100.0, annual_mu=0.07, annual_sigma=0.25)


@pytest.fixture(scope="module")
def paths() -> np.ndarray:
    return simulate_paths(PARAMS, SimulationConfig(days=126, paths=40_000, seed=11))


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def test_percentile_bands_are_ordered(paths: np.ndarray) -> None:
    p05, p50, p95 = percentile_bands(paths)

    assert p05.shape == (paths.shape[0],)
    assert np.all(p05 <= p50) and np.all(p50 <= p95)
    assert p05[0] == p50[0] == p95[0] == PARAMS.s0
    assert (p95 - p05)[-1] > (p95 - p05)[10]  # the fan widens with time


def test_expected_shortfall_is_at_least_var(paths: np.ndarray) -> None:
    terminal = paths[-1]
    for confidence in (0.90, 0.95, 0.99):
        assert expected_shortfall(terminal, PARAMS.s0, confidence) >= value_at_risk(
            terminal, PARAMS.s0, confidence
        )


def test_var_and_es_grow_with_confidence(paths: np.ndarray) -> None:
    terminal = paths[-1]
    var = [value_at_risk(terminal, PARAMS.s0, c) for c in (0.90, 0.95, 0.99)]
    es = [expected_shortfall(terminal, PARAMS.s0, c) for c in (0.90, 0.95, 0.99)]

    assert var == sorted(var)
    assert es == sorted(es)


def test_var_matches_lognormal_quantile(paths: np.ndarray) -> None:
    days = paths.shape[0] - 1
    m = PARAMS.log_drift * days
    s = PARAMS.sigma * math.sqrt(days)
    z05 = -1.6448536269514722
    expected_var = PARAMS.s0 - PARAMS.s0 * math.exp(m + s * z05)

    assert value_at_risk(paths[-1], PARAMS.s0, 0.95) == pytest.approx(expected_var, rel=0.03)


def test_var_on_a_known_distribution() -> None:
    terminal = np.arange(1, 101, dtype=float)  # losses from s0=100 are 0..99
    assert value_at_risk(terminal, 100.0, 0.95) == pytest.approx(94.05)
    assert expected_shortfall(terminal, 100.0, 0.95) == pytest.approx(97.0)


def test_target_probability_matches_closed_form(paths: np.ndarray) -> None:
    days = paths.shape[0] - 1
    target = 110.0
    d = (math.log(PARAMS.s0 / target) + PARAMS.log_drift * days) / (PARAMS.sigma * math.sqrt(days))

    assert probability_above(paths[-1], target) == pytest.approx(normal_cdf(d), abs=0.01)
    assert probability_above(paths[-1], target) + probability_below(paths[-1], target) == pytest.approx(1)


def test_summary_fields(paths: np.ndarray) -> None:
    summary = summarize(paths, confidence=0.95, target=120.0)
    data = summary.to_dict()

    assert summary.days == 126 and summary.paths == 40_000
    assert summary.p05 < summary.p50 < summary.p95
    assert summary.expected_shortfall >= summary.var
    assert data["var_pct"] == pytest.approx(summary.var / 100.0)
    assert data["prob_above_target"] + data["prob_below_target"] == pytest.approx(1)


def test_summary_without_target(paths: np.ndarray) -> None:
    summary = summarize(paths)

    assert summary.target is None and summary.prob_above_target is None


@pytest.mark.parametrize("confidence", [0, 1, 1.5])
def test_confidence_must_be_a_probability(paths: np.ndarray, confidence: float) -> None:
    with pytest.raises(ValueError):
        value_at_risk(paths[-1], PARAMS.s0, confidence)
