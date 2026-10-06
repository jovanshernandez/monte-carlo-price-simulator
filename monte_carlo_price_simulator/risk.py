"""Risk statistics computed from simulated price paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_PERCENTILES = (5.0, 50.0, 95.0)


def percentile_bands(
    paths: np.ndarray, percentiles: tuple[float, ...] = DEFAULT_PERCENTILES
) -> np.ndarray:
    """Per-day percentiles across paths. Returns shape ``(len(percentiles), days + 1)``."""
    return np.percentile(paths, percentiles, axis=1)


def probability_above(terminal: np.ndarray, target: float) -> float:
    return float(np.mean(terminal > target))


def probability_below(terminal: np.ndarray, target: float) -> float:
    return float(np.mean(terminal < target))


def value_at_risk(terminal: np.ndarray, s0: float, confidence: float = 0.95) -> float:
    """Loss per share that is not exceeded with probability ``confidence``.

    Loss is ``s0 - S_T``; VaR is the ``confidence`` quantile of that loss. A
    negative value means even the tail scenario finishes above the start price.
    """
    _check_confidence(confidence)
    losses = s0 - np.asarray(terminal, dtype=float)
    return float(np.quantile(losses, confidence))


def expected_shortfall(terminal: np.ndarray, s0: float, confidence: float = 0.95) -> float:
    """Average loss per share in the tail at or beyond VaR (also called CVaR)."""
    _check_confidence(confidence)
    losses = s0 - np.asarray(terminal, dtype=float)
    var = np.quantile(losses, confidence)
    return float(losses[losses >= var].mean())


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class RiskSummary:
    s0: float
    days: int
    paths: int
    confidence: float
    mean: float
    p05: float
    p50: float
    p95: float
    prob_above_start: float
    var: float
    expected_shortfall: float
    target: float | None = None
    prob_above_target: float | None = None
    prob_below_target: float | None = None

    @property
    def var_pct(self) -> float:
        return self.var / self.s0

    @property
    def es_pct(self) -> float:
        return self.expected_shortfall / self.s0

    def to_dict(self) -> dict[str, float | int | None]:
        data = asdict(self)
        data["var_pct"] = self.var_pct
        data["expected_shortfall_pct"] = self.es_pct
        return data


def summarize(
    paths: np.ndarray, confidence: float = 0.95, target: float | None = None
) -> RiskSummary:
    s0 = float(paths[0, 0])
    terminal = paths[-1]
    p05, p50, p95 = np.percentile(terminal, DEFAULT_PERCENTILES)
    return RiskSummary(
        s0=s0,
        days=paths.shape[0] - 1,
        paths=paths.shape[1],
        confidence=confidence,
        mean=float(terminal.mean()),
        p05=float(p05),
        p50=float(p50),
        p95=float(p95),
        prob_above_start=probability_above(terminal, s0),
        var=value_at_risk(terminal, s0, confidence),
        expected_shortfall=expected_shortfall(terminal, s0, confidence),
        target=target,
        prob_above_target=None if target is None else probability_above(terminal, target),
        prob_below_target=None if target is None else probability_below(terminal, target),
    )
