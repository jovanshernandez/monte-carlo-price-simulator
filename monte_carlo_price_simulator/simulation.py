"""Geometric Brownian motion: parameter estimation and vectorized path simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class GBMParams:
    """Daily GBM parameters.

    ``mu`` is the arithmetic drift (expected simple return per day) and ``sigma`` the
    daily volatility. Log returns are normal with mean ``mu - sigma**2 / 2`` and
    standard deviation ``sigma``.
    """

    s0: float
    mu: float
    sigma: float

    @property
    def log_drift(self) -> float:
        return self.mu - 0.5 * self.sigma**2

    @property
    def annual_mu(self) -> float:
        return self.mu * TRADING_DAYS_PER_YEAR

    @property
    def annual_sigma(self) -> float:
        return self.sigma * np.sqrt(TRADING_DAYS_PER_YEAR)

    @classmethod
    def from_annual(cls, s0: float, annual_mu: float, annual_sigma: float) -> GBMParams:
        return cls(
            s0=s0,
            mu=annual_mu / TRADING_DAYS_PER_YEAR,
            sigma=annual_sigma / np.sqrt(TRADING_DAYS_PER_YEAR),
        )


@dataclass(frozen=True)
class SimulationConfig:
    days: int = 252
    paths: int = 10_000
    seed: int | None = 42

    def validate(self) -> None:
        if self.days <= 0:
            raise ValueError("days must be positive")
        if self.paths <= 0:
            raise ValueError("paths must be positive")


def estimate_parameters(prices: pd.Series | np.ndarray) -> GBMParams:
    """Estimate daily GBM drift and volatility from a series of closing prices.

    Volatility is the sample standard deviation of daily log returns. The mean log
    return estimates ``mu - sigma**2 / 2``, so the arithmetic drift adds the
    variance term back.
    """
    values = pd.Series(prices).dropna().astype(float).to_numpy()
    if len(values) < 3:
        raise ValueError("at least three prices are required")
    if (values <= 0).any():
        raise ValueError("prices must be positive")

    log_returns = np.diff(np.log(values))
    sigma = float(log_returns.std(ddof=1))
    if sigma <= 0:
        raise ValueError("historical returns must have non-zero volatility")
    mu = float(log_returns.mean()) + 0.5 * sigma**2
    return GBMParams(s0=float(values[-1]), mu=mu, sigma=sigma)


def simulate_paths(params: GBMParams, config: SimulationConfig) -> np.ndarray:
    """Simulate GBM price paths.

    Returns an array of shape ``(days + 1, paths)``; row 0 is the starting price.
    The whole simulation is one draw of standard normals, a cumulative sum of log
    increments and one ``exp``, with no Python loop over days or paths.
    """
    config.validate()
    if params.s0 <= 0:
        raise ValueError("starting price must be positive")
    if params.sigma < 0:
        raise ValueError("volatility must be non-negative")

    rng = np.random.default_rng(config.seed)
    shocks = rng.standard_normal((config.days, config.paths))
    log_increments = params.log_drift + params.sigma * shocks

    log_paths = np.zeros((config.days + 1, config.paths))
    np.cumsum(log_increments, axis=0, out=log_paths[1:])
    return params.s0 * np.exp(log_paths)
