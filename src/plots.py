import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import lane_rpm_resid

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EQUIP_COLORS = {"Dry Van": "#2a78d6", "Reefer": "#eb6834", "Flatbed": "#1baf7a"}
MAIN = "#2a78d6"

plt.rcParams.update({
    "axes.edgecolor": "#9a9893", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _fig(w=8, h=3.6):
    return plt.subplots(figsize=(w, h))


def target_dist(tr):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.hist(tr["posted_rate"], bins=120, color=MAIN)
    a1.set(title="Posted rate", xlabel="$", ylabel="loads")
    rpm = tr["posted_rate"] / tr["distance"]
    a2.hist(rpm.clip(0, 6), bins=120, color=MAIN)
    a2.set(title="Rate per mile (clipped at 6)", xlabel="$/mi", yscale="log")
    fig.tight_layout()
    return fig


def rpm_vs_distance(tr, n=6000, seed=0):
    fig, ax = _fig()
    s = tr.sample(n, random_state=seed)
    for eq, c in EQUIP_COLORS.items():
        d = s[s["equipment"] == eq]
        ax.scatter(d["distance"], d["posted_rate"] / d["distance"], s=8, alpha=0.35, color=c, label=eq, linewidths=0)
    ax.set(title="Rate per mile falls w/ distance", xlabel="distance (mi)", ylabel="$/mi", ylim=(0, 6))
    ax.legend(frameon=False, markerscale=2)
    return fig


def monthly_trend(tr):
    from .trend import LevelTrend
    daily = LevelTrend("level").fit(tr).daily
    days = pd.Timestamp("2025-01-01") + pd.to_timedelta(daily.index, "D")
    g = pd.DataFrame({"r": daily.groupby(days.to_period("M")).mean()})
    g["mi"] = tr.groupby(tr["date"].dt.to_period("M"))["market_index"].mean()
    g.index = g.index.to_timestamp()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(g.index, (np.exp(g["r"]) - 1) * 100, color=MAIN, marker="o", markersize=5)
    a1.axhline(0, color="#9a9893", lw=0.8)
    a1.set(title="Rate level vs lane norm (%)", ylabel="%")
    a2.plot(g.index, g["mi"], color=MAIN, marker="o", markersize=5)
    a2.set(title="market_index (monthly mean)")
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def equipment_effect(tr):
    fig, ax = _fig(6, 3.2)
    rpm = (tr["posted_rate"] / tr["distance"]).groupby(tr["equipment"]).median().reindex(list(EQUIP_COLORS))
    ax.bar(rpm.index, rpm.values, color=[EQUIP_COLORS[e] for e in rpm.index], width=0.55)
    for i, v in enumerate(rpm.values):
        ax.text(i, v + 0.03, f"${v:.2f}", ha="center", color=INK2)
    ax.set(title="Median rate per mile by equipment", ylabel="$/mi", ylim=(0, rpm.max() * 1.2))
    return fig


def outliers(tr):
    fig, ax = _fig()
    r = lane_rpm_resid(tr)
    ax.hist(np.exp(r).clip(0, 7), bins=200, color=MAIN)
    ax.set(title="Rate / lane median rate (outlier clusters at ~0.3x and ~3.5x)", xlabel="ratio", ylabel="loads", yscale="log")
    return fig


def trend_fit(trend, title="Market level used by the model"):
    fig, ax = _fig()
    t0 = pd.Timestamp("2025-01-01")
    daily = trend.daily
    ax.scatter(t0 + pd.to_timedelta(daily.index, "D"), (np.exp(daily.values) - 1) * 100, s=6, color="#9a9893",
               label="daily mean vs lane norm")
    xs = t0 + pd.to_timedelta(np.arange(len(trend.smooth)), "D")
    ax.plot(xs, (np.exp(trend.smooth) - 1) * 100, color=MAIN, label="smoothed (in-sample)")
    fut = pd.date_range(xs[-1] + pd.Timedelta(days=1), "2025-12-31")
    ax.plot(fut, np.full(len(fut), (np.exp(trend.level) - 1) * 100), color=MAIN, ls="--", label="projected (last 60d level)")
    ax.set(title=title, ylabel="%")
    ax.legend(frameon=False)
    return fig


def feature_importance(fi):
    fig, ax = _fig(6.5, 3.6)
    fi = fi.sort_values("mae_increase")
    ax.barh(fi["feature"], fi["mae_increase"], color=MAIN, height=0.6)
    ax.set(title="Permutation importance (holdout MAE increase, $)", xlabel="$")
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    return fig
