#!/usr/bin/env python3
"""R3-V 全量消融实验 Macro-F1 图表（四个独立面板）。

数据: save/ablations_r3v_20261026/<arm>/<dataset>/<class>/seed_<s>/metrics.json
口径: 先 class 内对 3 seed 取平均, 再跨 class 等权平均; Δ = Full - Ablation;
      CI 为 class 级配对 bootstrap 95% (5000 次)。
输出: figure/ablation_f1_r3v_20261026/fig_{a,b,c,d}_*.{png,pdf}
用法: python scripts/tools/plot_ablation_f1.py          # 彩色版
      python scripts/tools/plot_ablation_f1.py --ieee   # IEEE 单/双栏版
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "save", "ablations_r3v_20261026")
OUT_DIR = os.path.join(ROOT, "figure", "ablation_f1_r3v_20261026")

DATASETS = [
    ("DDOS_19", "DDOS-19"),
    ("IDS2017", "CICIDS2017"),
    ("UNSW_NB15", "UNSW-NB15"),
    ("IoT-23", "IoT-23"),
    ("Kitsune", "Kitsune"),
    ("CSECICIDS2018", "CSE-CIC-IDS2018"),
]
ARMS = [
    "no_history_context",
    "no_retrieval",
    "no_gated_residual",
    "no_modal_attn",
    "no_pre_train",
]
ARM_LABEL = {
    "no_history_context": "w/o history context",
    "no_retrieval": "w/o graph retrieval",
    "no_gated_residual": "w/o gated residual",
    "no_modal_attn": "w/o cross-modal attn",
    "no_pre_train": "w/o pre-training",
}
ARM_TICK = {
    "no_history_context": "w/o\nhistory",
    "no_retrieval": "w/o\nretrieval",
    "no_gated_residual": "w/o\ngated res.",
    "no_modal_attn": "w/o\nmodal attn",
    "no_pre_train": "w/o\npre-train",
}
ARM_SHORT = {
    "no_history_context": "w/o history",
    "no_retrieval": "w/o retrieval",
    "no_gated_residual": "w/o gated res.",
    "no_modal_attn": "w/o modal attn",
    "no_pre_train": "w/o pre-train",
}
IEEE = "--ieee" in sys.argv
if IEEE:
    COLOR = {
        "full": "#000000",
        "no_history_context": "#D55E00",
        "no_retrieval": "#0072B2",
        "no_gated_residual": "#009E73",
        "no_modal_attn": "#CC79A7",
        "no_pre_train": "#E69F00",
    }
    HATCH = {
        "full": "",
        "no_history_context": "//",
        "no_retrieval": "\\\\",
        "no_gated_residual": ".",
        "no_modal_attn": "x",
        "no_pre_train": "--",
    }
    INK = "#000000"
    MUTED = "#333333"
    GRID = "#C8C8C8"
    CMAP = plt.get_cmap("RdBu")
else:
    COLOR = {
        "full": "#243B6B",
        "no_history_context": "#DFA23C",
        "no_retrieval": "#4E8FBF",
        "no_gated_residual": "#3E9E8A",
        "no_modal_attn": "#A97BB5",
        "no_pre_train": "#D4755E",
    }
    HATCH = {key: "" for key in ["full"] + ARMS}
    INK = "#2B2B2B"
    MUTED = "#6E6E6E"
    GRID = "#E4E4E4"
    CMAP = LinearSegmentedColormap.from_list(
        "soft_diverging",
        ["#A93E36", "#E7B3AB", "#F8F4EF", "#A9C4DE", "#2E5F8C"],
    )
METRIC = "macro_f1"


def load_rows():
    rows = defaultdict(lambda: defaultdict(list))
    pattern = os.path.join(SRC, "*", "*", "*", "seed_*", "metrics.json")
    for path in glob.glob(pattern):
        parts = path.split(os.sep)
        arm, dataset, cls = parts[-5], parts[-4], parts[-3]
        if arm not in ["full"] + ARMS:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        rows[dataset][(cls, arm)].append(payload)
    return rows


def mean_metric(entries, key=METRIC):
    vals = [e.get(key) for e in entries if e.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def bootstrap_ci(values, iterations=5000, seed=777):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(iterations, values.size))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def star(ci):
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return ""
    return "*" if (lo > 0 or hi < 0) else ""


def collect(rows):
    full = defaultdict(dict)
    arm_value = {}
    for dataset, _ in DATASETS:
        for (cls, arm), entries in rows.get(dataset, {}).items():
            value = mean_metric(entries)
            if value is None:
                continue
            arm_value[(dataset, cls, arm)] = value
            if arm == "full":
                full[dataset][cls] = value

    deltas = defaultdict(lambda: defaultdict(list))
    per_class = defaultdict(dict)
    for dataset, _ in DATASETS:
        for cls, full_value in full[dataset].items():
            for arm in ARMS:
                value = arm_value.get((dataset, cls, arm))
                if value is None:
                    continue
                delta = full_value - value
                deltas[dataset][arm].append(delta)
                per_class[dataset].setdefault(cls, {})[arm] = delta

    summary = defaultdict(dict)
    pooled = defaultdict(list)
    for dataset, _ in DATASETS:
        full_mean = float(np.mean(list(full[dataset].values())))
        for arm in ARMS:
            values = deltas[dataset][arm]
            summary[dataset][arm] = {
                "n": len(values),
                "full": full_mean,
                "d": float(np.mean(values)),
                "ci": bootstrap_ci(values),
                "win": int(np.sum(np.asarray(values) > 1e-6)),
            }
            pooled[arm].extend(values)

    pooled_summary = {}
    for arm in ARMS:
        values = pooled[arm]
        pooled_summary[arm] = {
            "n": len(values),
            "d": float(np.mean(values)),
            "ci": bootstrap_ci(values),
            "win": int(np.sum(np.asarray(values) > 1e-6)),
        }
    return full, arm_value, deltas, per_class, summary, pooled_summary


def setup_style():
    if IEEE:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#000000",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "hatch.linewidth": 0.5,
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#4A4A4A",
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#4A4A4A",
            "ytick.color": "#4A4A4A",
        })


def save(fig, stem):
    png = os.path.join(OUT_DIR, f"{stem}.png")
    pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.savefig(png, bbox_inches="tight", dpi=400)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("saved:", png)


def footer(fig, text, y=0.018):
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=7.0 if IEEE else 8.6, color=MUTED)


def clean_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle=":")
    ax.set_axisbelow(True)
    if IEEE:
        ax.tick_params(direction="in", top=True, right=True, length=2.5, width=0.6, pad=2)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3.2, width=0.8)


def draw_heatmap(ax, matrix, row_labels, col_labels, vmin, vmax, stars=None,
                 cbar_label="", title="", annot_size=8.4, cbar_fraction=0.038):
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    image = ax.imshow(matrix, cmap=CMAP, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=annot_size, color=MUTED)
                continue
            mark = stars[i][j] if stars is not None else ""
            clipped = abs(value) > max(abs(vmin), abs(vmax))
            color = "white" if abs(value) > 0.55 * max(abs(vmin), abs(vmax)) else INK
            ax.text(j, i, f"{value:+.3f}{mark}{'†' if clipped else ''}",
                    ha="center", va="center", fontsize=annot_size, color=color)

    ax.set_title(title, loc="left", pad=5 if IEEE else 10)
    cbar = plt.colorbar(image, ax=ax, fraction=cbar_fraction, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7.2 if IEEE else 9.2, color=INK)
    cbar.ax.tick_params(labelsize=6.8 if IEEE else 8.6, length=2 if IEEE else 3)
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor("#9A9A9A")


def fig_a(full, arm_value, summary):
    dataset = "DDOS_19"
    classes = sorted(full[dataset])
    arms_all = ["full"] + ARMS
    means, lo_err, hi_err, values_per_bar = [], [], [], []
    for arm in arms_all:
        if arm == "full":
            vals = [full[dataset][c] for c in classes]
        else:
            vals = [arm_value[(dataset, c, arm)] for c in classes]
        mu = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals)
        means.append(mu)
        lo_err.append(mu - lo)
        hi_err.append(hi - mu)
        values_per_bar.append(vals)

    fig, ax = plt.subplots(figsize=(3.5, 2.8) if IEEE else (7.6, 5.4))
    xs = np.arange(len(arms_all))
    bars = ax.bar(xs, means, width=0.62, color=[COLOR[a] for a in arms_all],
                  edgecolor="#1A1A1A" if IEEE else "#3A3A3A",
                  linewidth=0.5 if IEEE else 0.7,
                  alpha=1.0 if IEEE else 0.95, zorder=2)
    for patch, arm in zip(bars, arms_all):
        patch.set_hatch(HATCH[arm])
    ax.errorbar(xs, means, yerr=[lo_err, hi_err], fmt="none", ecolor="#1A1A1A" if IEEE else "#333333",
                elinewidth=0.8 if IEEE else 1.1,
                capsize=2.2 if IEEE else 3.5,
                capthick=0.8 if IEEE else 1.1, zorder=5)

    rng = np.random.default_rng(42)
    for x, vals in zip(xs, values_per_bar):
        ax.scatter(x + rng.normal(0, 0.055, len(vals)), vals,
                   s=6 if IEEE else 14,
                   facecolors="white",
                   edgecolors="#1A1A1A" if IEEE else "#3F3F3F",
                   linewidths=0.45 if IEEE else 0.6,
                   alpha=0.85 if IEEE else 0.9, zorder=4)

    ax.axhline(means[0], color=COLOR["full"],
               linestyle=(0, (4, 3)) if not IEEE else (0, (3, 2)),
               linewidth=0.8 if IEEE else 1.2, zorder=1)
    if IEEE:
        for idx, arm in enumerate(ARMS, start=1):
            item = summary[dataset][arm]
            ax.text(idx, 1.005, f"{item['d']:+.3f}{star(item['ci'])}", ha="center",
                    va="bottom", fontsize=6.2, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"Full\n({means[0]:.4f})"] + [ARM_TICK[a] for a in ARMS], fontsize=6.7)
        ax.set_ylim(0.28, 1.055)
        ax.set_ylabel("Macro-F1")
        ax.set_title("(a) DDOS-19", loc="left")
    else:
        ax.text(0, 1.008, f"Full = {means[0]:.4f}", ha="center", va="bottom",
                fontsize=9.0, color=COLOR["full"])
        for idx, arm in enumerate(ARMS, start=1):
            item = summary[dataset][arm]
            ax.text(idx, 1.008, f"Δ={item['d']:+.3f}{star(item['ci'])}", ha="center",
                    va="bottom", fontsize=9.0, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Full"] + [ARM_TICK[a] for a in ARMS])
        ax.set_ylim(0.28, 1.08)
        ax.set_ylabel("Macro-F1 (class-equal mean)")
        ax.set_title("(a) DDOS-19: Macro-F1 by ablation arm (17 classes × 3 seeds)", loc="left")

    clean_axes(ax)
    if IEEE:
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.23, top=0.92)
        footer(fig, "Δ = Full − Ablation; positive: Full better. Error bars: 95% class-level\n"
                    "paired bootstrap (5000 resamples). * = CI excludes 0. Axis truncated at 0.28.")
        save(fig, "fig_a_ddos19_macro_f1_ieee")
    else:
        fig.subplots_adjust(bottom=0.17)
        footer(fig, "Δ = Full − Ablation (positive: Full better).  Error bars: 95% class-level paired bootstrap\n"
                    "(5000 resamples); * = CI excludes 0.  Hollow dots: individual classes.  y-axis starts at 0.28.")
        save(fig, "fig_a_ddos19_macro_f1")


def fig_b(summary, pooled_summary):
    rows = [f"{name} ({summary[ds][ARMS[0]]['n']})" for ds, name in DATASETS]
    rows.append("All classes (50)")
    matrix = np.array([[summary[ds][arm]["d"] for arm in ARMS] for ds, _ in DATASETS])
    matrix = np.vstack([matrix, [pooled_summary[arm]["d"] for arm in ARMS]])
    stars = [[star(summary[ds][arm]["ci"]) for arm in ARMS] for ds, _ in DATASETS]
    stars.append([star(pooled_summary[arm]["ci"]) for arm in ARMS])

    fig, ax = plt.subplots(figsize=(6.95, 3.2) if IEEE else (8.8, 5.2))
    draw_heatmap(ax, matrix, rows, [ARM_SHORT[a] for a in ARMS], -0.08, 0.08, stars,
                 cbar_label="ΔMacro-F1  (Full − Ablation)",
                 title="(b) Dataset-level ablation effect" if IEEE
                 else "(b) Ablation effect by dataset (Δ > 0: Full better)",
                 annot_size=7.6 if IEEE else 8.8)
    if IEEE:
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.30, top=0.90)
        footer(fig, "* = 95% CI excludes 0.  † = cell colour clipped at ±0.08.\n"
                    "CSE-CIC-IDS2018 has one class, so no CI is available.")
        save(fig, "fig_b_dataset_heatmap_ieee")
    else:
        fig.subplots_adjust(left=0.17, bottom=0.22)
        footer(fig, "* = 95% CI excludes 0 (class-level paired bootstrap).  † = cell colour clipped at ±0.08.\n"
                    "CSE-CIC-IDS2018 has a single class, so no CI is available.")
        save(fig, "fig_b_dataset_heatmap")


def fig_c(deltas):
    data = [np.concatenate([np.asarray(deltas[ds][arm], dtype=float) for ds, _ in DATASETS])
            for arm in ARMS]
    fig, ax = plt.subplots(figsize=(3.5, 2.9) if IEEE else (8.2, 5.4))
    positions = np.arange(1, len(ARMS) + 1)
    boxes = ax.boxplot(data, positions=positions, widths=0.5 if not IEEE else 0.55, patch_artist=True,
                       showfliers=False, whis=1.5,
                       medianprops=dict(color="#1A1A1A", linewidth=1.2 if not IEEE else 0.8),
                       whiskerprops=dict(color="#6A6A6A" if not IEEE else "#1A1A1A",
                                         linewidth=1.0 if not IEEE else 0.6),
                       capprops=dict(color="#6A6A6A" if not IEEE else "#1A1A1A",
                                     linewidth=1.0 if not IEEE else 0.6),
                       boxprops=dict(linewidth=1.0 if not IEEE else 0.6))
    for patch, arm in zip(boxes["boxes"], ARMS):
        patch.set_facecolor(COLOR[arm])
        patch.set_alpha(0.28 if not IEEE else 0.34)
        patch.set_edgecolor(COLOR[arm] if not IEEE else "#1A1A1A")
        patch.set_hatch(HATCH[arm])

    rng = np.random.default_rng(7)
    ylim = (-0.29, 0.20)
    for pos, arm, values in zip(positions, ARMS, data):
        jitter = pos + rng.normal(0, 0.055, size=values.size)
        clipped = np.clip(values, ylim[0] + 0.005, 0.185)
        ax.scatter(jitter, clipped, s=7 if IEEE else 12, color=COLOR[arm],
                   alpha=0.62 if not IEEE else 0.70, linewidths=0, zorder=3)
        for xpos, value in zip(jitter, values):
            if value > 0.185:
                ax.annotate(f"{value:+.2f}", xy=(xpos, 0.185),
                            xytext=(xpos + 0.10, 0.185), ha="left", va="center",
                            fontsize=5.8 if IEEE else 7.4,
                            color=INK if IEEE else COLOR[arm],
                            arrowprops=dict(arrowstyle="-", color=COLOR[arm], lw=0.5 if IEEE else 0.6))
        ax.scatter(pos, float(np.mean(values)), marker="D", s=16 if IEEE else 26, facecolor="white",
                   edgecolor=COLOR[arm] if not IEEE else "#1A1A1A",
                   linewidth=0.8 if IEEE else 1.2, zorder=6)

    ax.axhline(0, color="#6A6A6A" if not IEEE else "#1A1A1A",
               linewidth=1.0 if not IEEE else 0.6, zorder=1)
    ax.set_xlim(0.45, len(ARMS) + 0.55)
    ax.set_ylim(*ylim)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{ARM_TICK[a]}\n({int(np.sum(v > 1e-6))}/{v.size})" for a, v in zip(ARMS, data)],
        fontsize=6.4 if IEEE else 8.8,
    )
    if IEEE:
        ax.set_ylabel("Per-class ΔMacro-F1", fontsize=8)
        ax.set_title("(c) Per-class effect (50 classes)", loc="left")
        clean_axes(ax)
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.29, top=0.91)
        footer(fig, "Δ = Full − Ablation; ◇ = mean; box = IQR; whiskers = 1.5×IQR.\n"
                    "Brackets: number of classes where Full is better.")
        save(fig, "fig_c_perclass_distribution_ieee")
    else:
        ax.set_ylabel("Per-class ΔMacro-F1")
        ax.set_title("(c) Per-class ablation effect across 50 classes", loc="left")
        clean_axes(ax)
        fig.subplots_adjust(bottom=0.20)
        footer(fig, "Δ = Full − Ablation.  Box: quartiles; whiskers: 1.5×IQR; ◇ = mean; dots = individual classes.\n"
                    "Brackets: classes where Full is better.  Values above +0.18 are annotated.")
        save(fig, "fig_c_perclass_distribution")


def fig_d(full, per_class):
    dataset = "DDOS_19"
    classes = sorted(full[dataset], key=lambda c: full[dataset][c])
    matrix = np.array([[per_class[dataset][c][arm] for arm in ARMS] for c in classes])
    labels = [f"{c}  ({full[dataset][c]:.2f})" for c in classes]
    fig, ax = plt.subplots(figsize=(6.95, 5.2) if IEEE else (8.6, 7.8))
    draw_heatmap(ax, matrix, labels, [ARM_SHORT[a] for a in ARMS], -0.06, 0.06,
                 cbar_label="ΔMacro-F1  (Full − Ablation)",
                 title="(d) DDOS-19 per-class detail" if IEEE
                 else "(d) DDOS-19 per-class ablation detail (Full Macro-F1 in brackets)",
                 annot_size=7.2 if IEEE else 8.4)
    if IEEE:
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.16, top=0.91)
        footer(fig, "Δ = Full − Ablation.  † = cell colour clipped at ±0.06\n"
                    "(UDP: +0.542).  Rows sorted by Full Macro-F1.")
        save(fig, "fig_d_ddos19_perclass_heatmap_ieee")
    else:
        fig.subplots_adjust(left=0.20, bottom=0.13)
        footer(fig, "Δ = Full − Ablation.  † = cell colour clipped at ±0.06 (UDP reaches +0.542).\n"
                    "Rows sorted by Full Macro-F1 (hardest on top).")
        save(fig, "fig_d_ddos19_perclass_heatmap")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()
    rows = load_rows()
    full, arm_value, deltas, per_class, summary, pooled_summary = collect(rows)
    fig_a(full, arm_value, summary)
    fig_b(summary, pooled_summary)
    fig_c(deltas)
    fig_d(full, per_class)

    print("\nDataset-level ΔMacro-F1 (Full − Ablation):")
    print("dataset".ljust(15) + "".join(arm.ljust(32) for arm in ARMS))
    for ds, _ in DATASETS:
        line = ds.ljust(15)
        for arm in ARMS:
            item = summary[ds][arm]
            line += f"{item['d']:+.4f} [{item['ci'][0]:+.4f},{item['ci'][1]:+.4f}]{star(item['ci'])}".ljust(32)
        print(line)


if __name__ == "__main__":
    main()
