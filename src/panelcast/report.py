"""Build reports/RESULTS.md and README figures (light + dark variants) from the gold/eval tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from panelcast.config import Settings  # noqa: E402
from panelcast.store import TableStore  # noqa: E402

log = logging.getLogger(__name__)

# Reference data-viz palette (validated categorical slots 1-2, blue<->red diverging pair) and chrome tokens.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "neg": "#e34948",
        "dim": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "s1": "#3987e5",
        "s2": "#d95926",
        "neg": "#e66767",
        "dim": "#4a4a46",
    },
}
MODEL_NAMES = {
    "naive": "Naive: last quarter's growth",
    "panel_raw": "Raw panel spend (OLS)",
    "panel_raked": "Raked panel + ticker effects",
    "ridge": "Ridge (multivariate)",
    "gbm": "Gradient boosting",
    "ensemble": "Ensemble (ridge + GBM)",
}
STEP_NAMES = {
    "raw": "Raw panel spend",
    "per_member": "Per enrolled member",
    "clean": "+ anomaly handling",
    "raked": "+ raking to census (final)",
}
ASSET_NAMES = {
    "card_panel_raked": "Card panel (raked)",
    "card_panel_raw": "Card panel (raw)",
    "wikipedia_pageviews": "Wikipedia pageviews (real)",
}


def _style(ax, t: dict, grid_axis: str = "x") -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _title(fig, t: dict, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, color=t["ink"], fontsize=12.5, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.905, subtitle, color=t["ink2"], fontsize=9.5, ha="left", va="top")


def _bar(ax, y: float, v: float, h: float, color: str, x_per_px: float, y_per_px: float) -> None:
    """Horizontal bar with a 4px rounded data end and a square end at the zero baseline."""
    r = 4 * x_per_px
    left, width = (0.0, v) if v > 0 else (v, -v)
    if width <= 2 * r:
        ax.add_patch(Rectangle((left, y - h / 2), width, h, color=color, linewidth=0))
        return
    ax.add_patch(
        FancyBboxPatch(
            (left, y - h / 2),
            width,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            mutation_aspect=y_per_px / x_per_px,
            color=color,
            linewidth=0,
        )
    )
    ax.add_patch(Rectangle((0.0 if v > 0 else -r, y - h / 2), r, h, color=color, linewidth=0))  # square baseline


def _rounded_hbars(ax, labels: list[str], values: list[float], colors: list[str], t: dict, fmt: str) -> None:
    """Single-series horizontal bars, first label on top, value labels at the bar tips."""
    ys = np.arange(len(values))[::-1]
    span = max(abs(v) for v in values) or 1
    ax.set_xlim(min(0, min(values)) - 0.3 * span * (min(values) < 0), max(0, max(values)) + 0.2 * span)
    ax.set_ylim(-0.7, len(values) - 0.3)
    fig = ax.get_figure()
    box = ax.get_position()
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (fig.get_size_inches()[0] * fig.dpi * box.width)
    y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (fig.get_size_inches()[1] * fig.dpi * box.height)
    for y, v, c in zip(ys, values, colors, strict=True):
        _bar(ax, y, v, 0.5, c, x_per_px, y_per_px)
        ha, off = ("left", 0.012 * span) if v >= 0 else ("right", -0.012 * span)
        ax.text(v + off, y, fmt.format(v), va="center", ha=ha, color=t["ink"], fontsize=9.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=t["ink2"], fontsize=9.5)
    ax.axvline(0, color=t["axis"], linewidth=1)


def _save(fig, out: Path, name: str, mode: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}_{mode}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def figure_panel_accuracy(panel: dict, out: Path) -> None:
    order = ["raw", "per_member", "clean", "raked"]
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.1), facecolor=t["surface"])
        fig.subplots_adjust(left=0.27, right=0.97, top=0.78, bottom=0.12)
        _style(ax, t)
        vals = [panel[m]["mae_pp"] for m in order]
        colors = [t["dim"]] * 3 + [t["s1"]]
        _rounded_hbars(ax, [STEP_NAMES[m] for m in order], vals, colors, t, "{:.1f} pp")
        ax.set_xlabel(
            "Mean absolute error of fiscal-quarter YoY growth vs. true spend (pp)", color=t["muted"], fontsize=9
        )
        _title(
            fig,
            t,
            "Each correction step cuts panel error",
            "Panel spend growth vs. the simulator's true card-visible spend, 20 tickers, 2019-2026",
        )
        _save(fig, out, "panel_accuracy", mode)


def figure_model_mae(metrics: pd.DataFrame, out: Path) -> None:
    m = metrics[metrics["segment"] == "all"].sort_values("mae_pp", ascending=True)
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.5), facecolor=t["surface"])
        fig.subplots_adjust(left=0.31, right=0.97, top=0.8, bottom=0.12)
        _style(ax, t)
        colors = [t["s1"] if x == "ensemble" else t["dim"] for x in m["model"]]
        _rounded_hbars(ax, [MODEL_NAMES[x] for x in m["model"]], list(m["mae_pp"]), colors, t, "{:.1f} pp")
        ax.set_xlabel("Mean absolute error of revenue YoY growth (pp)", color=t["muted"], fontsize=9)
        n = int(m["n"].iloc[0])
        _title(
            fig,
            t,
            "Nowcast error by model (lower is better)",
            f"Point-in-time walk-forward backtest vs. reported revenue, {n} ticker-quarters, 2020-2026",
        )
        _save(fig, out, "nowcast_mae", mode)


def figure_timeseries(preds: pd.DataFrame, tickers: list[str], out: Path) -> None:
    p = preds[preds["model"] == "ensemble"].copy()
    p["period_end"] = pd.to_datetime(p["period_end"])
    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, len(tickers), figsize=(10, 3.7), facecolor=t["surface"], sharey=False)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.66, bottom=0.12, wspace=0.28)
        for ax, ticker in zip(axes, tickers, strict=True):
            g = p[p["ticker"] == ticker].sort_values("period_end")
            _style(ax, t, grid_axis="y")
            actual = 100 * (np.exp(g["rev_yoy"]) - 1)
            nowcast = 100 * (np.exp(g["pred_yoy"]) - 1)
            ax.axhline(0, color=t["axis"], linewidth=1)
            ax.plot(g["period_end"], actual, color=t["s1"], linewidth=2, solid_capstyle="round")
            ax.plot(g["period_end"], nowcast, color=t["s2"], linewidth=2, solid_capstyle="round")
            for series, color in ((actual, t["s1"]), (nowcast, t["s2"])):
                ax.scatter(
                    g["period_end"], series, s=22, color=color, edgecolors=t["surface"], linewidths=1.2, zorder=3
                )
            ax.set_title(ticker, color=t["ink"], fontsize=10.5, loc="left", fontweight="bold")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.tick_params(axis="x", labelsize=8.5)
            for label in ax.get_xticklabels():
                label.set_color(t["ink2"])
        handles = [
            plt.Line2D([], [], color=t["s1"], linewidth=2, marker="o", markersize=5, label="Reported revenue growth"),
            plt.Line2D(
                [],
                [],
                color=t["s2"],
                linewidth=2,
                marker="o",
                markersize=5,
                label="Ensemble nowcast (made 7 days after quarter end)",
            ),
        ]
        leg = fig.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.86), ncol=2, frameon=False, fontsize=9
        )
        for text in leg.get_texts():
            text.set_color(t["ink2"])
        _title(
            fig,
            t,
            "Nowcasts track reported growth through COVID and after",
            "Fiscal-quarter YoY revenue growth; each nowcast uses only data available before the filing",
        )
        _save(fig, out, "nowcast_timeseries", mode)


def figure_assets(card: pd.DataFrame, out: Path) -> None:
    c = card.set_index("asset").loc[list(ASSET_NAMES)]
    vals = list(100 * c["oos_skill_vs_naive"])
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 2.7), facecolor=t["surface"])
        fig.subplots_adjust(left=0.29, right=0.95, top=0.74, bottom=0.17)
        _style(ax, t)
        colors = [t["s1"] if v >= 0 else t["neg"] for v in vals]
        _rounded_hbars(ax, [ASSET_NAMES[a] for a in c.index], vals, colors, t, "{:+.0f}%")
        ax.set_xlabel("Out-of-sample error reduction vs. naive forecast (%)", color=t["muted"], fontsize=9)
        _title(
            fig,
            t,
            "Which data asset improves revenue nowcasts?",
            "Walk-forward skill = 1 - MAE(naive + asset) / MAE(naive); negative means the asset hurts",
        )
        _save(fig, out, "asset_skill", mode)


def _table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, float) and not np.isnan(v) else ("" if isinstance(v, float) else str(v))
            for v in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(settings: Settings, store: TableStore) -> None:
    root = settings.reports_dir
    figs = root / "figures"
    ev = json.loads((root / "evaluation.json").read_text())
    metrics = store.read_pandas("gold", "nowcast_metrics")
    preds = store.read_pandas("gold", "nowcast_predictions")
    card = store.read_pandas("gold", "asset_scorecard").sort_values("oos_skill_vs_naive", ascending=False)
    events = store.read_pandas("gold", "anomaly_events").sort_values("start")

    figure_panel_accuracy(ev["panel_estimators"], figs)
    figure_model_mae(metrics, figs)
    figure_timeseries(preds, ["WMT", "SBUX", "UBER"], figs)
    figure_assets(card, figs)

    m_all = metrics[metrics["segment"] == "all"].sort_values("mae_pp")
    seg = metrics.pivot(index="model", columns="segment", values="mae_pp").loc[m_all["model"]]
    by_ticker = (
        preds[preds["model"].isin(["naive", "ensemble"])]
        .assign(abs_err=lambda d: d["error_pp"].abs())
        .pivot_table(index="ticker", columns="model", values="abs_err", aggfunc="mean")
        .sort_values("ensemble")
    )
    er = ev["entity_resolution"]
    lines = [
        "# PanelCast results",
        "",
        "Generated by `panelcast run report`. Card transactions are simulated; revenue, fiscal calendars and "
        "Wikipedia pageviews are real.",
        "",
        "## Revenue nowcast backtest",
        "",
        _table(
            m_all[
                [
                    "model",
                    "n",
                    "mae_pp",
                    "median_ae_pp",
                    "mape_revenue",
                    "direction_acc",
                    "beats_naive",
                    "interval_coverage",
                ]
            ].round(3),
            "{:.3f}",
        ),
        "",
        "MAE by period (pp):",
        "",
        _table(seg.reset_index().round(2)),
        "",
        "Per-ticker MAE, ensemble vs. naive (pp):",
        "",
        _table(by_ticker.reset_index().round(2)),
        "",
        "## Panel estimators vs. ground truth",
        "",
        _table(pd.DataFrame(ev["panel_estimators"]).T.reset_index(names="estimator").round(2)),
        "",
        "## Entity resolution vs. ground truth",
        "",
        _table(pd.DataFrame(er).T.reset_index(names="level").round(4), "{:.4f}"),
        "",
        "## Data-quality anomalies detected",
        "",
        f"Detected {ev['anomaly_detection']['detected']} of {ev['anomaly_detection']['injected']} injected anomalies, "
        f"{ev['anomaly_detection']['false_positives']} false positives, median |delay| "
        f"{ev['anomaly_detection']['median_abs_delay_days']:.0f} days.",
        "",
        _table(
            events.drop(
                columns=[
                    c
                    for c in events.columns
                    if c not in ("type", "source", "ticker", "start", "end", "n_days", "action")
                ]
            )
        ),
        "",
        "## Data quality (silver)",
        "",
        "```json",
        json.dumps(ev["data_quality"], indent=2),
        "```",
        "",
        "## Alternative-data asset scorecard",
        "",
        _table(card.drop(columns=["quality"]).round(3), "{:.3f}"),
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s and figures in %s", root / "RESULTS.md", figs)
