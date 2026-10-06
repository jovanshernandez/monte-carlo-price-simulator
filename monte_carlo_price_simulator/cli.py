"""Command-line entry point: ``mc-sim``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from monte_carlo_price_simulator.market_data import fetch_close_prices, load_csv_prices
from monte_carlo_price_simulator.risk import RiskSummary, percentile_bands, summarize
from monte_carlo_price_simulator.simulation import (
    GBMParams,
    SimulationConfig,
    estimate_parameters,
    simulate_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mc-sim",
        description="Simulate future prices with geometric Brownian motion and report "
        "percentile bands, target probabilities, Value-at-Risk and Expected Shortfall.",
    )
    parser.add_argument("ticker", nargs="?", help="ticker to download from Yahoo Finance")
    parser.add_argument("--csv", type=Path, help="read closes from a CSV instead (no network)")
    parser.add_argument("--lookback-days", type=int, default=365,
                        help="calendar days of history used for estimation (default: 365)")
    parser.add_argument("--days", type=int, default=252,
                        help="trading days to simulate (default: 252)")
    parser.add_argument("--paths", type=int, default=10_000,
                        help="number of simulated paths (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--target", type=float, help="price target for above/below probabilities")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="VaR / ES confidence level (default: 0.95)")
    parser.add_argument("--drift", type=float,
                        help="override annual drift, e.g. 0 for a zero-drift risk view")
    parser.add_argument("--vol", type=float, help="override annual volatility, e.g. 0.25")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                        help="where reports and charts are written (default: reports/)")
    parser.add_argument("--no-plots", action="store_true", help="skip the PNG charts")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(os.environ.get("FORCE_COLOR")) or sys.stdout.isatty()


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def head(self, text: str) -> str:
        return self._wrap("1;36", text)

    def dim(self, text: str) -> str:
        return self._wrap("90", text)

    def bad(self, text: str) -> str:
        return self._wrap("31", text)

    def good(self, text: str) -> str:
        return self._wrap("32", text)


def format_report(
    label: str, prices: pd.Series, params: GBMParams, config: SimulationConfig,
    summary: RiskSummary, written: list[Path], color: bool = False,
) -> str:
    s = _Style(color)
    conf = f"{summary.confidence:.0%}"
    if isinstance(prices.index, pd.DatetimeIndex):
        span = f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d} ({len(prices)} closes)"
    else:
        span = f"{len(prices)} closes"

    def row(name: str, value: str, width: int = 26) -> str:
        return f"  {name:<{width}}{value}"

    def usd(value: float) -> str:
        return f"{'$' + format(value, ',.2f'):>11}"

    lines = [
        s.head(f"Monte Carlo price simulation: {label}"),
        row("History", span, 14),
        row("Model", f"GBM, drift {params.annual_mu:+.1%}/yr, vol {params.annual_sigma:.1%}/yr", 14),
        row("Simulation", f"{config.paths:,} paths x {config.days} trading days, seed {config.seed}", 14),
        "",
        s.head(f"Terminal price after {summary.days} trading days"),
        row("Start", usd(summary.s0)),
        row("Mean", usd(summary.mean)),
        row("5th percentile", usd(summary.p05)),
        row("Median", usd(summary.p50)),
        row("95th percentile", usd(summary.p95)),
        "",
        s.head("Probabilities"),
        row("P(finish above start)", f"{summary.prob_above_start:>11.1%}"),
    ]
    if summary.target is not None:
        lines += [
            row(f"P(finish above ${summary.target:,.2f})", s.good(f"{summary.prob_above_target:>11.1%}")),
            row(f"P(finish below ${summary.target:,.2f})", s.bad(f"{summary.prob_below_target:>11.1%}")),
        ]
    lines += [
        "",
        s.head(f"Risk over the horizon ({conf} confidence, per share)"),
        row("Value-at-Risk", s.bad(usd(summary.var)) + f"  ({summary.var_pct:.1%} of start)"),
        row("Expected Shortfall", s.bad(usd(summary.expected_shortfall))
            + f"  ({summary.es_pct:.1%} of start)"),
    ]
    if written:
        folder = written[0].parent
        names = ", ".join(p.name if p.parent == folder else str(p) for p in written)
        lines += ["", s.dim(f"Wrote to {folder}/: {names}")]
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> RiskSummary:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.ticker and not args.csv:
        parser.error("give a ticker or --csv PATH")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")

    if args.csv:
        prices = load_csv_prices(args.csv, args.lookback_days)
        label = (args.ticker or args.csv.stem).upper()
    else:
        prices = fetch_close_prices(args.ticker, args.lookback_days)
        label = args.ticker.upper()

    params = estimate_parameters(prices)
    if args.drift is not None or args.vol is not None:
        params = GBMParams.from_annual(
            params.s0,
            params.annual_mu if args.drift is None else args.drift,
            params.annual_sigma if args.vol is None else args.vol,
        )
    config = SimulationConfig(days=args.days, paths=args.paths, seed=args.seed)
    paths = simulate_paths(params, config)
    summary = summarize(paths, confidence=args.confidence, target=args.target)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = label.replace("/", "_")
    summary_path = out / f"{stem}_summary.json"
    bands_path = out / f"{stem}_bands.csv"

    payload = {
        "label": label,
        "model": {"annual_drift": params.annual_mu, "annual_vol": params.annual_sigma,
                  "daily_mu": params.mu, "daily_sigma": params.sigma},
        "simulation": {"days": config.days, "paths": config.paths, "seed": config.seed},
        "summary": summary.to_dict(),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    p05, p50, p95 = percentile_bands(paths)
    pd.DataFrame({"day": np.arange(config.days + 1), "p05": p05, "p50": p50, "p95": p95}).to_csv(
        bands_path, index=False, float_format="%.4f"
    )
    written = [summary_path, bands_path]

    if not args.no_plots:
        from monte_carlo_price_simulator.plotting import save_fan_chart, save_terminal_histogram

        history = prices if isinstance(prices.index, pd.DatetimeIndex) else None
        written.append(save_fan_chart(paths, summary, out / f"{stem}_fan.png", label, history))
        written.append(save_terminal_histogram(paths, summary, out / f"{stem}_terminal.png", label))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(label, prices, params, config, summary, written, _color_enabled()))
    return summary


def main() -> None:
    try:
        run()
    except ValueError as exc:
        print(f"mc-sim: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
