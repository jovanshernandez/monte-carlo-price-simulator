"""Charts: a fan chart of simulated paths and a terminal-price histogram with VaR."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter, StrMethodFormatter  # noqa: E402

from monte_carlo_price_simulator.risk import RiskSummary, percentile_bands  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
PATH_GRAY = "#9a9994"
BLUE = "#2a78d6"
RED = "#e34948"
ORANGE = "#eb6834"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "font.size": 10.5,
    "text.parse_math": False,  # prices contain "$"
}

PRICE_FORMAT = StrMethodFormatter("${x:,.0f}")


def _forecast_index(history: pd.Series | None, days: int) -> pd.Index:
    if history is not None and isinstance(history.index, pd.DatetimeIndex) and len(history):
        start = history.index[-1]
        return pd.DatetimeIndex([start]).append(
            pd.bdate_range(start + pd.offsets.BDay(1), periods=days)
        )
    return pd.RangeIndex(days + 1)


def save_fan_chart(
    paths: np.ndarray,
    summary: RiskSummary,
    output: Path,
    label: str,
    history: pd.Series | None = None,
    history_days: int = 252,
    sample_paths: int = 60,
) -> Path:
    days = paths.shape[0] - 1
    x = _forecast_index(history, days)
    p05, p25, p50, p75, p95 = percentile_bands(paths, (5, 25, 50, 75, 95))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6), dpi=110)

        if history is not None and len(history):
            recent = history.iloc[-history_days:]
            hx = recent.index if isinstance(x, pd.DatetimeIndex) else np.arange(-len(recent) + 1, 1)
            ax.plot(hx, recent.to_numpy(), color=INK, linewidth=1.6, label="History")

        n = min(sample_paths, paths.shape[1])
        ax.plot(x, paths[:, :n], color=PATH_GRAY, linewidth=0.5, alpha=0.35)
        ax.fill_between(x, p05, p95, color=BLUE, alpha=0.14, linewidth=0, label="5th-95th percentile")
        ax.fill_between(x, p25, p75, color=BLUE, alpha=0.24, linewidth=0, label="25th-75th percentile")
        ax.plot(x, p50, color=BLUE, linewidth=2, label="Median path")

        ax.axhline(summary.s0, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        if summary.target is not None:
            ax.axhline(summary.target, color=ORANGE, linewidth=1.5, linestyle=(0, (6, 3)),
                       label=f"Target ${summary.target:,.2f}")

        end = x[-1]
        for value, name in ((p95[-1], "95th"), (p50[-1], "median"), (p05[-1], "5th")):
            ax.annotate(f"{name}  ${value:,.2f}", xy=(end, value), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9.5, color=INK)

        ax.set_title(f"{label}: {paths.shape[1]:,} GBM paths over {days} trading days")
        ax.set_ylabel("Price")
        ax.yaxis.set_major_formatter(PRICE_FORMAT)
        ax.margins(x=0)
        ax.set_xlim(right=_pad_right(ax, x))
        ax.legend(loc="upper left", ncols=2)
        fig.text(0.01, 0.01, f"Start ${summary.s0:,.2f}  |  dashed line marks the start price",
                 fontsize=9, color=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
    return output


def _pad_right(ax, x) -> object:
    """Leave room after the last forecast day for the percentile labels."""
    left, right = ax.get_xlim()
    return right + (right - left) * 0.11


def save_terminal_histogram(
    paths: np.ndarray, summary: RiskSummary, output: Path, label: str
) -> Path:
    terminal = paths[-1]
    var_price = summary.s0 - summary.var
    es_price = summary.s0 - summary.expected_shortfall
    conf = f"{summary.confidence:.0%}"

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
        counts, edges = np.histogram(terminal, bins=90)
        share = counts / counts.sum()
        centers = (edges[:-1] + edges[1:]) / 2
        width = (edges[1] - edges[0]) * 0.86
        colors = np.where(centers <= var_price, RED, BLUE)
        ax.bar(centers, share, width=width, color=colors, alpha=0.85, linewidth=0)

        top = share.max()
        lines = [
            (summary.s0, INK_SECONDARY, (0, (4, 3)), f"Start ${summary.s0:,.2f}"),
            (var_price, RED, "-", f"VaR {conf}: -${summary.var:,.2f} ({summary.var_pct:.1%})"),
            (es_price, RED, (0, (2, 2)), f"ES {conf}: -${summary.expected_shortfall:,.2f} "
                                          f"({summary.es_pct:.1%})"),
        ]
        if summary.target is not None:
            lines.append((summary.target, ORANGE, (0, (6, 3)),
                          f"Target ${summary.target:,.2f}: P(above) {summary.prob_above_target:.1%}"))
        for value, color, style, text in lines:
            ax.axvline(value, color=color, linewidth=1.5, linestyle=style, label=text)

        ax.text(0.99, 0.97,
                f"Median ${summary.p50:,.2f}\n5th-95th ${summary.p05:,.2f} - ${summary.p95:,.2f}\n"
                f"P(finish above start) {summary.prob_above_start:.1%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
                linespacing=1.6)

        ax.set_title(f"{label}: terminal price after {summary.days} trading days "
                     f"({summary.paths:,} paths)")
        ax.set_xlabel("Terminal price")
        ax.set_ylabel("Share of paths")
        ax.set_ylim(0, top * 1.12)
        ax.xaxis.set_major_formatter(PRICE_FORMAT)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax.legend(loc="upper left")
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
    return output
