import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle


# =========================================================
# Global style
# =========================================================
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

MAIN_CMAP = "viridis"
DROP_CMAP = "plasma"
SCORE_CMAP = "cividis"


# =========================================================
# Basic helpers
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def savefig(fig, out: Path):
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def normalize_curve(obj):
    if isinstance(obj, list):
        return obj
    return []


def p_to_text(p):
    if np.isnan(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3g}"


def add_spearman_box(ax, x, y, extra_text="", loc="lower right"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return

    rho, p = spearmanr(x[mask], y[mask])

    text = f"Spearman ρ = {rho:.3f}\np = {p_to_text(p)}"
    if extra_text:
        text += f"\n{extra_text}"

    # Determine position and alignment based on loc parameter
    pos_map = {
        "lower right": (0.97, 0.03, "top", "right"),
        "lower left": (0.03, 0.03, "top", "left"),
        "upper right": (0.97, 0.97, "top", "right"),
        "upper left": (0.03, 0.97, "top", "left"),
    }
    
    x_pos, y_pos, va, ha = pos_map.get(loc, (0.97, 0.03, "top", "right"))

    ax.text(
        x_pos, y_pos, text,
        transform=ax.transAxes,
        va=va,
        ha=ha,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
        zorder=20,
    )


def add_linear_trend(ax, x, y, label="Overall trend", color="black", linestyle="--", linewidth=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None

    coef = np.polyfit(x[mask], y[mask], 1)
    xx = np.linspace(np.min(x[mask]), np.max(x[mask]), 300)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, linestyle=linestyle, color=color, linewidth=linewidth, label=label, zorder=2)
    return coef


def annotate_subset(ax, df, xcol, ycol, text_func, max_labels=6, fontsize=8, dx=4, dy=4):
    if df.empty:
        return

    count = 0
    for _, row in df.iterrows():
        if count >= max_labels:
            break

        x = row[xcol]
        y = row[ycol]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        txt = text_func(row)
        if not txt:
            continue

        ax.annotate(
            txt,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            color="black",
        )
        count += 1


# =========================================================
# Pareto helpers
# =========================================================
def pareto_frontier_maximize_x_y(df, xcol, ycol):
    """
    maximize x, maximize y
    """
    pts = df[[xcol, ycol]].dropna()
    if pts.empty:
        return df.iloc[[]].copy()

    keep_idx = []
    for i, ri in pts.iterrows():
        xi, yi = ri[xcol], ri[ycol]
        dominated = False
        for j, rj in pts.iterrows():
            if i == j:
                continue
            xj, yj = rj[xcol], rj[ycol]
            if (xj >= xi and yj >= yi) and (xj > xi or yj > yi):
                dominated = True
                break
        if not dominated:
            keep_idx.append(i)

    return df.loc[keep_idx].copy().sort_values(xcol)


def pareto_frontier_maximize_x_minimize_y(df, xcol, ycol):
    """
    maximize x, minimize y
    """
    pts = df[[xcol, ycol]].dropna()
    if pts.empty:
        return df.iloc[[]].copy()

    keep_idx = []
    for i, ri in pts.iterrows():
        xi, yi = ri[xcol], ri[ycol]
        dominated = False
        for j, rj in pts.iterrows():
            if i == j:
                continue
            xj, yj = rj[xcol], rj[ycol]
            if (xj >= xi and yj <= yi) and (xj > xi or yj < yi):
                dominated = True
                break
        if not dominated:
            keep_idx.append(i)

    return df.loc[keep_idx].copy().sort_values(xcol)


# =========================================================
# JSON parser
# =========================================================
def parse_json_to_df(json_path: Path, lambda_score: float = 2.0) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    experiments = obj.get("experiments", [])
    configs = obj.get("configurations_by_experiment", [])
    max_accs = obj.get("max_test_accuracy_by_experiment", [])
    summaries = obj.get("summaries_by_experiment", [])

    rows = []

    for i, exp in enumerate(experiments):
        exp = exp or {}

        cfg = exp.get("configuration") or (configs[i] if i < len(configs) else {}) or {}
        summ = exp.get("final_summary_physics_aware") or (summaries[i] if i < len(summaries) else {}) or {}
        max_info = exp.get("max_test_accuracy") or (max_accs[i] if i < len(max_accs) else {}) or {}

        pack_ratio = safe_float(cfg.get("pack_ratio", np.nan))
        global_sparsity = safe_float(cfg.get("global_sparsity", np.nan))

        source_log = str(exp.get("source_log", "") or "")
        is_baseline = ("baseline" in source_log.lower()) or (pack_ratio == 0.0)

        test_acc_curve = exp.get("test_accuracies_percent") or []
        test_loss_curve = exp.get("test_losses") or []
        train_loss_curve = exp.get("train_losses") or []
        epochs = exp.get("epochs") or []

        if isinstance(epochs, list) and len(epochs) > 0 and isinstance(epochs[0], dict):
            ep_list, ta, tl, tr = [], [], [], []
            for item in epochs:
                item = item or {}
                ep_list.append(item.get("epoch", len(ep_list) + 1))
                ta.append(safe_float(item.get("test_acc", np.nan)))
                tl.append(safe_float(item.get("test_loss", np.nan)))
                tr.append(safe_float(item.get("train_loss", np.nan)))
            epochs = ep_list
            if len(test_acc_curve) == 0:
                test_acc_curve = ta
            if len(test_loss_curve) == 0:
                test_loss_curve = tl
            if len(train_loss_curve) == 0:
                train_loss_curve = tr

        max_acc = safe_float(
            max_info.get("max_test_accuracy_percent", np.nan)
            if isinstance(max_info, dict) else np.nan
        )

        final_acc = safe_float(test_acc_curve[-1] if len(test_acc_curve) else np.nan)
        last5_mean_acc = safe_float(
            np.mean(test_acc_curve[-5:]) if len(test_acc_curve) >= 5
            else (np.mean(test_acc_curve) if len(test_acc_curve) > 0 else np.nan)
        )
        final_test_loss = safe_float(test_loss_curve[-1] if len(test_loss_curve) else np.nan)

        rows.append({
            "pack_ratio": pack_ratio,
            "global_sparsity": global_sparsity,
            "is_baseline": is_baseline,
            "source_log": source_log,
            "max_acc": max_acc,
            "final_acc": final_acc,
            "last5_mean_acc": last5_mean_acc,
            "final_test_loss": final_test_loss,
            "slice_reduction": safe_float(summ.get("slice_reduction_percent", np.nan) if isinstance(summ, dict) else np.nan),
            "pin_reduction": safe_float(summ.get("pin_reduction_rate_percent", np.nan) if isinstance(summ, dict) else np.nan),
            "total_dead": safe_float(summ.get("total_dead", np.nan) if isinstance(summ, dict) else np.nan),
            "successful_packs": safe_float(summ.get("successful_packs_pairs", np.nan) if isinstance(summ, dict) else np.nan),
            "failed_packs": safe_float(summ.get("failed_packs_pairs", np.nan) if isinstance(summ, dict) else np.nan),
            "epochs": epochs if isinstance(epochs, list) else [],
            "test_acc_curve": normalize_curve(test_acc_curve),
            "test_loss_curve": normalize_curve(test_loss_curve),
            "train_loss_curve": normalize_curve(train_loss_curve),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    baseline_df = df[df["is_baseline"]].copy()

    if not baseline_df.empty:
        baseline_acc = baseline_df["max_acc"].dropna().max()
        baseline_loss = baseline_df["final_test_loss"].dropna().min() if baseline_df["final_test_loss"].notna().any() else np.nan
    else:
        baseline_acc = np.nan
        baseline_loss = np.nan

    df["baseline_acc_global"] = baseline_acc
    df["baseline_loss_global"] = baseline_loss

    # positive accuracy drop only
    raw_drop = baseline_acc - df["max_acc"]
    df["acc_drop_positive"] = np.maximum(0.0, raw_drop)

    df["loss_delta_vs_global_baseline"] = df["final_test_loss"] - baseline_loss

    total_pairs = df["successful_packs"].fillna(0) + df["failed_packs"].fillna(0)
    df["pack_efficiency"] = np.where(total_pairs > 0, df["successful_packs"] / total_pairs, np.nan)

    # practical tradeoff score
    df["tradeoff_score"] = df["pin_reduction"] - lambda_score * df["acc_drop_positive"]

    return df.reset_index(drop=True)


# =========================================================
# Zoom helper
# =========================================================
def add_zoom_inset(ax, df, xcol, ycol, xlim, ylim, label_col=None, loc="lower left", frontier_mode="maxmax"):
    axins = inset_axes(ax, width="38%", height="38%", loc=loc, borderpad=2)

    zdf = df.dropna(subset=[xcol, ycol]).copy()

    axins.scatter(
        zdf[xcol], zdf[ycol],
        s=18,
        alpha=0.35,
        color="#8aa76b",
        zorder=1
    )

    nobase = zdf[~zdf["is_baseline"]].copy()
    if frontier_mode == "maxmin":
        pf_zoom = pareto_frontier_maximize_x_minimize_y(nobase, xcol, ycol)
    else:
        pf_zoom = pareto_frontier_maximize_x_y(nobase, xcol, ycol)

    if not pf_zoom.empty:
        axins.plot(
            pf_zoom[xcol], pf_zoom[ycol],
            "-o",
            color="#4c9a2a",
            linewidth=2.0,
            markersize=4,
            zorder=3
        )

    b_zoom = zdf[zdf["is_baseline"]].copy()
    if not b_zoom.empty:
        b_zoom = b_zoom.sort_values(xcol)
        axins.plot(
            b_zoom[xcol], b_zoom[ycol],
            "-X",
            color="#4e73a8",
            linewidth=1.8,
            markersize=5,
            zorder=2
        )

    xv = pd.to_numeric(zdf[xcol], errors="coerce")
    yv = pd.to_numeric(zdf[ycol], errors="coerce")
    mask = np.isfinite(xv) & np.isfinite(yv)
    if mask.sum() >= 2:
        coef = np.polyfit(xv[mask], yv[mask], 1)
        xx = np.linspace(np.min(xv[mask]), np.max(xv[mask]), 200)
        yy = coef[0] * xx + coef[1]
        axins.plot(xx, yy, "--", color="black", linewidth=1.2, zorder=2)

    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_title("Zoom", fontsize=9)
    axins.grid(True, alpha=0.25)

    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        linewidth=1.2,
        edgecolor="black",
        linestyle="--",
        facecolor="none"
    )
    ax.add_patch(rect)

    if label_col is not None:
        inside = zdf[
            (zdf[xcol] >= xlim[0]) & (zdf[xcol] <= xlim[1]) &
            (zdf[ycol] >= ylim[0]) & (zdf[ycol] <= ylim[1])
        ].copy()

        if frontier_mode == "maxmin":
            inside_frontier = pareto_frontier_maximize_x_minimize_y(inside[~inside["is_baseline"]].copy(), xcol, ycol)
        else:
            inside_frontier = pareto_frontier_maximize_x_y(inside[~inside["is_baseline"]].copy(), xcol, ycol)

        label_candidates = inside_frontier if not inside_frontier.empty else inside.head(4)

        for _, row in label_candidates.iterrows():
            val = row[label_col]
            if pd.notna(val):
                axins.text(
                    row[xcol], row[ycol],
                    f"{val:.1f}" if isinstance(val, (int, float, np.floating)) else str(val),
                    fontsize=6
                )

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35")
    return axins


# =========================================================
# Figures
# =========================================================
def plot_main_tradeoff_panels(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pin_reduction", "max_acc", "acc_drop_positive", "pack_ratio", "global_sparsity"])
    if d.empty:
        return

    baseline = df[df["is_baseline"]].copy().dropna(subset=["pin_reduction", "max_acc"])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15, wspace=0.35)

    # (a) max acc vs pin reduction
    ax = axes[0]
    sc1 = ax.scatter(
        d["pin_reduction"], d["max_acc"],
        c=d["pack_ratio"],
        s=35 + 110 * d["global_sparsity"],
        cmap=MAIN_CMAP,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
        label="All configs"
    )
    add_linear_trend(ax, d["pin_reduction"], d["max_acc"], label="Overall trend")

    pf = pareto_frontier_maximize_x_y(d, "pin_reduction", "max_acc")
    if not pf.empty:
        ax.plot(
            pf["pin_reduction"], pf["max_acc"],
            "-o", color="#4c9a2a", linewidth=2.2, markersize=5,
            label="Pareto frontier", zorder=5
        )
        annotate_subset(
            ax, pf.head(6), "pin_reduction", "max_acc",
            lambda r: f"PR={r['pack_ratio']:.1f}",
            max_labels=6
        )

    if not baseline.empty:
        baseline = baseline.sort_values("pin_reduction")
        ax.plot(
            baseline["pin_reduction"], baseline["max_acc"],
            "-X", color="#4e73a8", linewidth=2.0, markersize=8,
            label="Baseline (PR=0)", zorder=4
        )

    add_spearman_box(
        ax, d["pin_reduction"], d["max_acc"], loc="lower left"
    )
    ax.set_title("(a) Maximum Accuracy vs Pin Reduction")
    ax.set_xlabel("Pin Reduction Rate (%)")
    ax.set_ylabel("Maximum Test Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    zoom_df = df.dropna(subset=["pin_reduction", "max_acc"]).copy()
    add_zoom_inset(
        ax, zoom_df, "pin_reduction", "max_acc",
        xlim=(68, 86),
        ylim=(55.5, 59.0),
        label_col="pack_ratio",
        loc="lower left",
        frontier_mode="maxmax"
    )

    cbar1 = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label("Packing ratio")

    # (b) positive acc drop vs pin reduction
    ax = axes[1]
    sc2 = ax.scatter(
        d["pin_reduction"], d["acc_drop_positive"],
        c=d["pack_ratio"],
        s=35 + 110 * d["global_sparsity"],
        cmap=DROP_CMAP,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.35,
        zorder=3
    )
    add_linear_trend(ax, d["pin_reduction"], d["acc_drop_positive"], label="Overall trend")

    pf2 = pareto_frontier_maximize_x_minimize_y(d, "pin_reduction", "acc_drop_positive")
    if not pf2.empty:
        ax.plot(
            pf2["pin_reduction"], pf2["acc_drop_positive"],
            "-o", color="#2c7c31", linewidth=2.2, markersize=5,
            label="Tradeoff frontier", zorder=5
        )
        annotate_subset(
            ax, pf2.head(6), "pin_reduction", "acc_drop_positive",
            lambda r: f"PR={r['pack_ratio']:.1f}",
            max_labels=6
        )

    add_spearman_box(
        ax, d["pin_reduction"], d["acc_drop_positive"], loc="lower right"
    )
    ax.set_title("(b) Accuracy Penalty vs Pin Reduction")
    ax.set_xlabel("Pin Reduction Rate (%)")
    ax.set_ylabel("Accuracy Drop from Baseline (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    zoom_df2 = df.dropna(subset=["pin_reduction", "acc_drop_positive"]).copy()
    add_zoom_inset(
        ax, zoom_df2, "pin_reduction", "acc_drop_positive",
        xlim=(58, 82),
        ylim=(-0.2, 2.2),
        label_col="pack_ratio",
        loc="upper left",
        frontier_mode="maxmin"
    )

    cbar2 = fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Packing ratio")
    fig.tight_layout(pad=1.2)

    savefig(fig, out)


def plot_accuracy_drop_vs_pin(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pin_reduction", "acc_drop_positive", "pack_ratio", "global_sparsity"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 7.0))

    sc = ax.scatter(
        d["pin_reduction"], d["acc_drop_positive"],
        c=d["pack_ratio"],
        s=35 + 110 * d["global_sparsity"],
        cmap=DROP_CMAP,
        alpha=0.92,
        edgecolor="black",
        linewidth=0.35,
        zorder=3
    )

    add_linear_trend(ax, d["pin_reduction"], d["acc_drop_positive"], label="Overall trend")

    pf = pareto_frontier_maximize_x_minimize_y(d, "pin_reduction", "acc_drop_positive")
    if not pf.empty:
        ax.plot(
            pf["pin_reduction"], pf["acc_drop_positive"],
            "-o", color="#2c7c31", linewidth=2.2, markersize=5,
            label="Tradeoff frontier", zorder=5
        )
        annotate_subset(
            ax, pf.head(6), "pin_reduction", "acc_drop_positive",
            lambda r: f"PR={r['pack_ratio']:.1f}",
            max_labels=6
        )

    add_spearman_box(
        ax, d["pin_reduction"], d["acc_drop_positive"], loc="lower left"
    )
    ax.set_title("L1: Accuracy Drop vs Pin Reduction")
    ax.set_xlabel("Pin Reduction Rate (%)")
    ax.set_ylabel("Accuracy Drop from Baseline (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Packing ratio")
    fig.tight_layout(pad=1.0)

    savefig(fig, out)


def plot_hardware_consistency(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pin_reduction", "slice_reduction", "pack_ratio", "global_sparsity"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    sc = ax.scatter(
        d["pin_reduction"], d["slice_reduction"],
        c=d["pack_ratio"],
        s=35 + 110 * d["global_sparsity"],
        cmap=MAIN_CMAP,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.35
    )

    add_linear_trend(ax, d["pin_reduction"], d["slice_reduction"], label="Overall trend")
    add_spearman_box(
        ax, d["pin_reduction"], d["slice_reduction"], loc="lower left"
    )

    ax.set_title("L1: Pin Reduction vs Slice Reduction")
    ax.set_xlabel("Pin Reduction Rate (%)")
    ax.set_ylabel("Slice Reduction (%)")
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Packing ratio")
    fig.tight_layout(pad=1.0)

    savefig(fig, out)


def plot_slice_reduction_vs_acc_drop(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["slice_reduction", "acc_drop_positive", "pack_ratio", "global_sparsity"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.7, 7.0))

    sc = ax.scatter(
        d["slice_reduction"], d["acc_drop_positive"],
        c=d["pack_ratio"],
        s=35 + 110 * d["global_sparsity"],
        cmap=DROP_CMAP,
        alpha=0.92,
        edgecolor="black",
        linewidth=0.35,
        zorder=3
    )

    add_linear_trend(ax, d["slice_reduction"], d["acc_drop_positive"], label="Overall trend")

    pf = pareto_frontier_maximize_x_minimize_y(d, "slice_reduction", "acc_drop_positive")
    if not pf.empty:
        ax.plot(
            pf["slice_reduction"], pf["acc_drop_positive"],
            "-o", color="#2c7c31", linewidth=2.2, markersize=5,
            label="Tradeoff frontier", zorder=5
        )

    add_spearman_box(
        ax, d["slice_reduction"], d["acc_drop_positive"], loc="lower left"
    )

    ax.set_title("L1: Slice Reduction vs Accuracy Drop")
    ax.set_xlabel("Slice Reduction (%)")
    ax.set_ylabel("Accuracy Drop from Baseline (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Packing ratio")

    savefig(fig, out)


def plot_best_per_pack_ratio(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pack_ratio", "max_acc", "pin_reduction", "global_sparsity"])
    if d.empty:
        return

    best_rows = []
    for pr, sub in d.groupby("pack_ratio"):
        sub = sub.sort_values(["max_acc", "pin_reduction"], ascending=[False, False])
        best_rows.append(sub.iloc[0])

    best_df = pd.DataFrame(best_rows).sort_values("pack_ratio")
    if best_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.5))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15, wspace=0.3)

    ax = axes[0]
    ax.plot(best_df["pack_ratio"], best_df["max_acc"], "-o", linewidth=2.0, markersize=6, label="Best per PR")
    add_linear_trend(ax, best_df["pack_ratio"], best_df["max_acc"], label="Linear trend")
    add_spearman_box(
        ax, best_df["pack_ratio"], best_df["max_acc"], loc="lower right"
    )
    annotate_subset(
        ax, best_df, "pack_ratio", "max_acc",
        lambda r: f"GS={r['global_sparsity']:.1f}",
        max_labels=6
    )
    ax.set_title("(a) Best Accuracy by Packing Ratio")
    ax.set_xlabel("Packing Ratio")
    ax.set_ylabel("Best Maximum Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.plot(best_df["pack_ratio"], best_df["pin_reduction"], "-o", linewidth=2.0, markersize=6, label="Best-acc config")
    add_linear_trend(ax, best_df["pack_ratio"], best_df["pin_reduction"], label="Linear trend")
    add_spearman_box(
        ax, best_df["pack_ratio"], best_df["pin_reduction"], loc="lower left"
    )
    annotate_subset(
        ax, best_df, "pack_ratio", "pin_reduction",
        lambda r: f"GS={r['global_sparsity']:.1f}",
        max_labels=6
    )
    ax.set_title("(b) Hardware Gain at Best Accuracy")
    ax.set_xlabel("Packing Ratio")
    ax.set_ylabel("Pin Reduction Rate (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    savefig(fig, out)


def plot_grouped_acc_vs_global_sparsity(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["global_sparsity", "max_acc", "pack_ratio"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(9.1, 6.8))

    prs = sorted([x for x in d["pack_ratio"].unique() if np.isfinite(x)])
    has_any = False
    for pr in prs:
        sub = d[np.isclose(d["pack_ratio"], pr)].sort_values("global_sparsity")
        if len(sub) < 2:
            continue
        has_any = True
        ax.plot(
            sub["global_sparsity"], sub["max_acc"],
            "-o", linewidth=1.8, markersize=4,
            label=f"PR={pr:.1f}"
        )

    if not has_any:
        return

    add_linear_trend(ax, d["global_sparsity"], d["max_acc"], label="Overall trend")
    add_spearman_box(ax, d["global_sparsity"], d["max_acc"], loc="lower right")

    ax.set_title("L1: Maximum Accuracy vs Global Sparsity")
    ax.set_xlabel("Global Sparsity")
    ax.set_ylabel("Maximum Test Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout(pad=1.0)

    savefig(fig, out)


def plot_grouped_acc_vs_pack_ratio(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pack_ratio", "max_acc", "global_sparsity"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(9.1, 6.8))

    gss = sorted([x for x in d["global_sparsity"].unique() if np.isfinite(x)])
    has_any = False
    for gs in gss:
        sub = d[np.isclose(d["global_sparsity"], gs)].sort_values("pack_ratio")
        if len(sub) < 2:
            continue
        has_any = True
        ax.plot(
            sub["pack_ratio"], sub["max_acc"],
            "-o", linewidth=1.8, markersize=4,
            label=f"GS={gs:.1f}"
        )

    if not has_any:
        return

    add_linear_trend(ax, d["pack_ratio"], d["max_acc"], label="Overall trend")
    add_spearman_box(ax, d["pack_ratio"], d["max_acc"], loc="lower right")

    ax.set_title("L1: Maximum Accuracy vs Packing Ratio")
    ax.set_xlabel("Packing Ratio")
    ax.set_ylabel("Maximum Test Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout(pad=1.0)

    savefig(fig, out)


def plot_last5_errorbars(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pack_ratio", "pin_reduction", "last5_mean_acc"])
    if d.empty:
        return

    grouped = d.groupby(["pack_ratio", "pin_reduction"], as_index=False).agg(
        mean_last5=("last5_mean_acc", "mean"),
        std_last5=("last5_mean_acc", "std"),
        n=("last5_mean_acc", "count")
    )
    if grouped.empty:
        return

    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    prs = sorted([x for x in grouped["pack_ratio"].unique() if np.isfinite(x)])
    for pr in prs:
        sub = grouped[np.isclose(grouped["pack_ratio"], pr)].sort_values("pin_reduction")
        if sub.empty:
            continue

        ax.errorbar(
            sub["pin_reduction"], sub["mean_last5"],
            yerr=sub["std_last5"].fillna(0).values,
            fmt="-o",
            capsize=3,
            linewidth=1.6,
            markersize=4,
            label=f"PR={pr:.1f}"
        )

    add_spearman_box(
        ax, grouped["pin_reduction"], grouped["mean_last5"], loc="lower right"
    )
    ax.set_title("L1: Last-5 Mean Accuracy vs Pin Reduction")
    ax.set_xlabel("Pin Reduction Rate (%)")
    ax.set_ylabel("Last-5 Mean Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout(pad=1.0)

    savefig(fig, out)


def plot_pareto_frontier(df: pd.DataFrame, out: Path):
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pin_reduction", "max_acc"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10.2, 7.6))

    ax.scatter(
        d["pin_reduction"], d["max_acc"],
        s=24,
        alpha=0.25,
        color="#9fba71",
        label="All configs",
        zorder=1
    )

    pf = pareto_frontier_maximize_x_y(d, "pin_reduction", "max_acc")
    if not pf.empty:
        ax.plot(
            pf["pin_reduction"], pf["max_acc"],
            "-o", color="#4c9a2a", linewidth=2.3, markersize=5,
            label="Pareto frontier", zorder=3
        )
        annotate_subset(
            ax, pf.head(6), "pin_reduction", "max_acc",
            lambda r: f"PR={r['pack_ratio']:.1f}" if pd.notna(r["pack_ratio"]) else "",
            max_labels=6
        )

    b = df[df["is_baseline"]].copy().dropna(subset=["pin_reduction", "max_acc"])
    if not b.empty:
        b = b.sort_values("pin_reduction")
        ax.plot(
            b["pin_reduction"], b["max_acc"],
            "-X", color="#4e73a8", linewidth=2.1, markersize=8,
            label="Baseline", zorder=2
        )

    add_spearman_box(ax, d["pin_reduction"], d["max_acc"], loc="upper right")

    ax.set_title("Pareto Frontier: L1 Maximum Accuracy vs Pin Reduction")
    ax.set_xlabel("Pin Reduction Rate (%)")
    ax.set_ylabel("Maximum Test Accuracy (%)")
    ax.grid(True, alpha=0.25)

    zoom_df = df.dropna(subset=["pin_reduction", "max_acc"]).copy()
    add_zoom_inset(
        ax, zoom_df, "pin_reduction", "max_acc",
        xlim=(68, 93),
        ylim=(49.5, 58.2),
        label_col="pack_ratio",
        loc="lower left",
        frontier_mode="maxmax"
    )

    ax.legend(loc="best")
    fig.tight_layout(pad=1.0)
    savefig(fig, out)


def plot_epoch_compare(df: pd.DataFrame, out: Path, pr: float, gs: float):
    d = df[~df["is_baseline"]].copy().dropna(subset=["pack_ratio", "global_sparsity"])
    b = df[df["is_baseline"]].copy()

    sub = d[np.isclose(d["pack_ratio"], pr) & np.isclose(d["global_sparsity"], gs)].copy()
    if sub.empty:
        return

    row = sub.iloc[0]
    acc = normalize_curve(row["test_acc_curve"])
    test_loss = normalize_curve(row["test_loss_curve"])
    epochs = normalize_curve(row["epochs"])

    if len(epochs) == 0:
        epochs = list(range(1, max(len(acc), len(test_loss)) + 1))

    base_acc = []
    base_loss = []
    base_epochs = []

    if not b.empty:
        brow = b.iloc[0]
        base_acc = normalize_curve(brow["test_acc_curve"])
        base_loss = normalize_curve(brow["test_loss_curve"])
        base_epochs = normalize_curve(brow["epochs"])
        if len(base_epochs) == 0:
            base_epochs = list(range(1, max(len(base_acc), len(base_loss)) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15, wspace=0.3)

    ax = axes[0]
    if len(acc) > 0:
        ax.plot(epochs[:len(acc)], acc, linewidth=1.9, label=f"Pruned (PR={pr}, GS={gs})")
    if len(base_acc) > 0:
        ax.plot(base_epochs[:len(base_acc)], base_acc, linewidth=1.8, linestyle="--", label="Baseline")

    if len(epochs) > 0:
        phase_epoch = int(0.68 * len(epochs))
        ax.axvline(phase_epoch, color="gray", linestyle=":", linewidth=1.4)
        ax.text(phase_epoch + 2, np.min(acc) if len(acc) else 0, "Recovery phase?", fontsize=8)

    ax.set_title(f"Accuracy vs Epoch (PR={pr}, GS={gs})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    if len(test_loss) > 0:
        ax.plot(epochs[:len(test_loss)], test_loss, linewidth=1.9, label=f"Pruned (PR={pr}, GS={gs})")
    if len(base_loss) > 0:
        ax.plot(base_epochs[:len(base_loss)], base_loss, linewidth=1.8, linestyle="--", label="Baseline")

    if len(epochs) > 0:
        phase_epoch = int(0.68 * len(epochs))
        ax.axvline(phase_epoch, color="gray", linestyle=":", linewidth=1.4)
        ax.text(phase_epoch + 2, np.min(test_loss) if len(test_loss) else 0, "Recovery phase?", fontsize=8)

    ax.set_title(f"Test Loss vs Epoch (PR={pr}, GS={gs})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout(pad=1.0)

    savefig(fig, out)


def plot_tradeoff_score(df: pd.DataFrame, out: Path, lambda_score: float):
    """
    score = pin_reduction - lambda * acc_drop_positive
    """
    d = df[~df["is_baseline"]].copy()
    d = d.dropna(subset=["pin_reduction", "max_acc", "tradeoff_score", "pack_ratio", "global_sparsity"])
    if d.empty:
        return

    topk = d.sort_values("tradeoff_score", ascending=False).head(5).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.15, wspace=0.35)

    # left: actual plane + top-score highlight
    ax = axes[0]
    sc = ax.scatter(
        d["pin_reduction"], d["max_acc"],
        c=d["tradeoff_score"],
        s=35 + 110 * d["global_sparsity"],
        cmap=SCORE_CMAP,
        alpha=0.92,
        edgecolor="black",
        linewidth=0.35,
        zorder=3
    )
    add_linear_trend(ax, d["pin_reduction"], d["max_acc"], label="Overall trend")

    ax.scatter(
        topk["pin_reduction"], topk["max_acc"],
        s=180,
        facecolors="none",
        edgecolors="red",
        linewidth=1.8,
        zorder=5,
        label="Top tradeoff score"
    )
    annotate_subset(
        ax, topk, "pin_reduction", "max_acc",
        lambda r: f"PR={r['pack_ratio']:.1f}, GS={r['global_sparsity']:.1f}",
        max_labels=5
    )

    ax.set_title(f"(a) Tradeoff Score Highlight (λ={lambda_score})")
    ax.set_xlabel("Pin Reduction Rate (%)")
    ax.set_ylabel("Maximum Test Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tradeoff score")

    # right: ranked bar plot
    ax = axes[1]
    ranked = d.sort_values("tradeoff_score", ascending=False).head(10).copy()
    labels = [f"PR={r.pack_ratio:.1f}\nGS={r.global_sparsity:.1f}" for _, r in ranked.iterrows()]
    ax.bar(range(len(ranked)), ranked["tradeoff_score"].values)
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels(labels)
    ax.set_title("(b) Top Configurations by Tradeoff Score")
    ax.set_ylabel("Tradeoff score")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(pad=1.0)

    savefig(fig, out)


def export_summary_csv(df: pd.DataFrame, out_csv: Path):
    d = df[~df["is_baseline"]].copy()

    cols = [
        "pack_ratio", "global_sparsity", "max_acc", "acc_drop_positive",
        "pin_reduction", "slice_reduction", "final_test_loss",
        "last5_mean_acc", "successful_packs", "failed_packs",
        "pack_efficiency", "tradeoff_score"
    ]
    keep = [c for c in cols if c in d.columns]
    d[keep].sort_values(["pack_ratio", "global_sparsity"]).to_csv(out_csv, index=False)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-json", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="l1_paper_figures_output")
    parser.add_argument("--best-pr", type=float, default=0.8)
    parser.add_argument("--best-gs", type=float, default=0.5)
    parser.add_argument("--lambda-score", type=float, default=2.0)
    args = parser.parse_args()

    json_path = Path(args.parsed_json)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = parse_json_to_df(json_path, lambda_score=args.lambda_score)
    if df.empty:
        print("No valid experiments found.")
        return

    df.to_csv(outdir / "l1_full_summary.csv", index=False)
    export_summary_csv(df, outdir / "l1_summary.csv")

    # Main figures
    plot_main_tradeoff_panels(df, outdir / "fig1_main_tradeoff_panels.png")
    plot_pareto_frontier(df, outdir / "fig2_pareto_max_acc_vs_pin_reduction.png")
    plot_hardware_consistency(df, outdir / "fig3_pin_reduction_vs_slice_reduction.png")

    # Important supporting figures
    plot_accuracy_drop_vs_pin(df, outdir / "fig4_accuracy_drop_vs_pin_reduction.png")
    plot_best_per_pack_ratio(df, outdir / "fig5_best_per_pack_ratio.png")
    plot_tradeoff_score(df, outdir / "fig6_tradeoff_score.png", lambda_score=args.lambda_score)

    # Supplementary figures
    plot_slice_reduction_vs_acc_drop(df, outdir / "supp_slice_reduction_vs_acc_drop.png")
    plot_grouped_acc_vs_global_sparsity(df, outdir / "supp_grouped_max_acc_vs_global_sparsity.png")
    plot_grouped_acc_vs_pack_ratio(df, outdir / "supp_grouped_max_acc_vs_pack_ratio.png")
    plot_last5_errorbars(df, outdir / "supp_last5_mean_acc_vs_pin_reduction_errorbars.png")
    plot_epoch_compare(
        df,
        outdir / f"supp_epoch_compare_pr{args.best_pr}_gs{args.best_gs}.png",
        args.best_pr,
        args.best_gs,
    )

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()