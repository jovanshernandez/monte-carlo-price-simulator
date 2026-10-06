"""Price history loaders: Yahoo Finance (network) or a local CSV (offline)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

CLOSE_COLUMNS = ("Adj Close", "Close", "adj_close", "close", "price")


def fetch_close_prices(ticker: str, lookback_days: int = 730) -> pd.Series:
    """Download daily adjusted closes from Yahoo Finance via yfinance."""
    if not ticker.strip():
        raise ValueError("ticker is required")
    if lookback_days <= 1:
        raise ValueError("lookback_days must be greater than 1")

    import yfinance as yf  # imported lazily so offline use never needs it

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    data = yf.download(
        ticker.upper(),
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if data is None or data.empty or "Close" not in data:
        raise ValueError(f"no close price data returned for {ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns one column per ticker
        close = close.iloc[:, 0]
    return close.dropna().astype(float).rename(ticker.upper())


def load_csv_prices(path: str | Path, lookback_days: int | None = None) -> pd.Series:
    """Load closing prices from a CSV with a date column and a close column.

    The first column is treated as the date. The close column is the first of
    ``Adj Close``, ``Close``, ``close`` or ``price`` that exists.
    """
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"{path}: expected a date column and a close column")
    column = next((c for c in CLOSE_COLUMNS if c in frame.columns), None)
    if column is None:
        raise ValueError(f"{path}: no close column found (tried {', '.join(CLOSE_COLUMNS)})")

    index = pd.to_datetime(frame.iloc[:, 0])
    prices = pd.Series(frame[column].to_numpy(dtype=float), index=index, name=Path(path).stem)
    prices = prices.dropna().sort_index()
    if lookback_days is not None:
        prices = prices[prices.index > prices.index[-1] - pd.Timedelta(days=lookback_days)]
    return prices
