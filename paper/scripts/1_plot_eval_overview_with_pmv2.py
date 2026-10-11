"""Composite figure for fig:locomo_eval_overview (UPDATED).

Layout: 2 rows x 4 columns, single figure.

  Top row (a)-(d): Backbone generalization on 4 datasets (existing data).
  Bottom row (e)-(f): NEW PMv2 baseline-comparison panels, one per context
                      window (32k, 128k). Each panel shows 6 systems x 2
                      formats (MCQ vs Open-Ended). PreCoMem is highlighted
                      with the deep scientific blue used throughout the paper.

We span the bottom 4 grid cells with 2 wider axes so the per-system bars
have enough room for value labels without growing the overall figure
height (still fits the previous backbone-overview footprint).
"""

from __future__ import annotations

import _style as st
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

# Display scale (same idea as 3_plot_frontier_memory_row.py / 2_plot_ablation_dimobench_row.py):
# raise/lower `_UP` to resize titles, ticks, legends, bar labels, and bracket layout.
_UP = 1.4

def _fs(x: float) -> int:
    return int(round(x * _UP))


FS_PANEL_TITLE = _fs(13.0)
TITLE_PAD_PT = 3.5 * _UP
TITLE_PAD_TOP_PT = TITLE_PAD_PT + 0.9 * _UP
TITLE_PAD_BOTTOM_PT = max(1.8 * _UP, TITLE_PAD_PT - 1.3 * _UP)
FS_BAR_VALUE = _fs(13.0)
FS_BAR_VALUE_TOP = _fs(12.0)
FS_BRACKET_DELTA = _fs(12.3)
FS_TICK = _fs(11.2)
FS_XTICK_TOP = _fs(10.0)  # two-line backbone labels in the four narrow top panels
FS_XTICK_BOTTOM = _fs(10.2)  # six method names per PMv2 panel
FS_YLABEL = _fs(13.0)
FS_FIG_LEGEND_ROW1 = _fs(11.8)
FS_FIG_LEGEND_ROW2 = _fs(11.5)
FS_BASE_RC = _fs(12)

# Bar-top label anchor offsets (data coordinates); scale with `_UP` when fonts grow.
BAR_LABEL_Y_BACKBONE = 1.6 * _UP
BAR_LABEL_Y_PMV2 = 1.1 * _UP

# Approximate vertical extent of bar-top numerics in **data coordinates** (large fonts).
LABEL_EM_BACKBONE = 4.2 * _UP
# PMv2: arms near value-label centerline (data coords).
PMV2_LABEL_EM_CENTER = 2.4 * _UP
PMV2_BRACKET_TOUCH_OFFSET = 0.0
PMV2_BRACKET_ARM_RISE = 0.78 * _UP
PMV2_DELTA_TEXT_OFFSET = 0.34 * _UP
PMV2_YLIM_HEADROOM = 3.0 * _UP

# Backbone Δ brackets (data coordinates).
BACKBONE_ARM_ABOVE_ERR = 2.4 * _UP
BACKBONE_BRACKET_GAP = 1.35 * _UP
BACKBONE_JOINT_PAD = 0.25 * _UP
BACKBONE_DELTA_TEXT_RISE = 0.62 * _UP
BRACKET_LINEWIDTH = max(0.75, 1.0 * min(_UP, 1.45))

# Figure-level legend frames (top / middle strips): light fill + dashed outline.
LEGEND_FRAME_FACE = "#ffffff"
LEGEND_FRAME_EDGE = "#888888"
LEGEND_FRAME_ALPHA = 0.92
LEGEND_FRAME_LINewidth = max(0.8, 0.95 * min(_UP, 1.45))


def _style_fig_legend_dashed_frame(leg: Legend) -> None:
    """Turn on the legend patch and draw a subtle dashed border for quicker scanning."""
    leg.set_frame_on(True)
    fr = leg.get_frame()
    fr.set_facecolor(LEGEND_FRAME_FACE)
    fr.set_edgecolor(LEGEND_FRAME_EDGE)
    fr.set_alpha(LEGEND_FRAME_ALPHA)
    fr.set_linewidth(LEGEND_FRAME_LINewidth)
    fr.set_linestyle("--")


BACKBONES = ["Qwen3.5-35B", "Qwen3.5-122B", "GPT-5-Chat"]
# Camera-ready: each x-axis group fixes the answer model (left name); the
# "+ PreCoMem" bar delegates memory construction to the one-tier-smaller
# backbone (right name) while answering with the same model.
BACKBONE_LABEL = {
    "Qwen3.5-35B":  "Qwen-35B\nmem\u2192\u200a9B",
    "Qwen3.5-122B": "Qwen-122B\nmem\u2192\u200a35B",
    "GPT-5-Chat":   "GPT-5-Chat\nmem\u2192\u200amini",
}
DATASETS = ["LoCoMo", "LongMemEval", "PMv2 (32k)", "PMv2 (128k)"]
TOP_TITLES = ["(a) LoCoMo", "(b) LongMemEval-S",
              "(c) PersonaMem-v2 (32k)", "(d) PersonaMem-v2 (128k)"]

PMV2_METHODS = ["MemOS", "MemBox", "O-Mem", "TiMem", "EverMemOS", "PreCoMem"]
PMV2_LABEL = {m: m for m in PMV2_METHODS}
PMV2_LABEL["PreCoMem"] = "PreCoMem\n(Ours)"


# ------------------------- Top row: backbone gen ------------------------- #

def _load_backbone_data():
    rows = st.read_csv("backbone_generalization.csv")
    out = {d: {} for d in DATASETS}
    for r in rows:
        out[r["dataset"]][r["backbone"]] = (
            float(r["wo_PreCoMem"]),
            float(r["wo_std"]),
            float(r["with_PreCoMem"]),
            float(r["with_std"]),
        )
    return out


def draw_backbone_panel(ax, data, title):
    x = np.arange(len(BACKBONES))
    width = 0.36
    wo_vals = [data[b][0] for b in BACKBONES]
    wo_stds = [data[b][1] for b in BACKBONES]
    wi_vals = [data[b][2] for b in BACKBONES]
    wi_stds = [data[b][3] for b in BACKBONES]

    err_kwargs = dict(capsize=2.4, elinewidth=0.9, ecolor="#333")
    ax.bar(x - width / 2, wo_vals, width, yerr=wo_stds,
           color=st.SCI_BLUE_MID, edgecolor="#333", linewidth=0.6,
           error_kw=err_kwargs)
    ax.bar(x + width / 2, wi_vals, width, yerr=wi_stds,
           color=st.SCI_BLUE_DARK, edgecolor="#333", linewidth=0.6,
           hatch="//", error_kw=err_kwargs)

    # Labels are nudged outward from the group centre so the wider text never
    # overlaps the neighbouring (taller) bar.
    lbl_dx = 0.09
    for xi, (wo, wo_s, wi, wi_s) in enumerate(zip(wo_vals, wo_stds, wi_vals, wi_stds)):
        ax.text(xi - width / 2 - lbl_dx, wo + wo_s + BAR_LABEL_Y_BACKBONE, f"{wo:.1f}",
                ha="center", va="bottom", fontsize=FS_BAR_VALUE_TOP, color="#333")
        ax.text(xi + width / 2 + lbl_dx, wi + wi_s + BAR_LABEL_Y_BACKBONE, f"{wi:.1f}",
                ha="center", va="bottom", fontsize=FS_BAR_VALUE_TOP,
                color=st.SCI_BLUE_DARK, fontweight="bold")

    bracket_color = st.ACCENT_RED
    for xi, (wo, wo_s, wi, wi_s) in enumerate(zip(wo_vals, wo_stds, wi_vals, wi_stds)):
        # Compute the gain from the rounded bar labels so the bracket matches what the reader sees.
        delta = round(wi, 1) - round(wo, 1)
        if delta <= 0:
            continue
        # Joint above error bars; lift past bar numerics so arms / bracket never clip text.
        label_top_wo = wo + wo_s + BAR_LABEL_Y_BACKBONE + LABEL_EM_BACKBONE
        label_top_wi = wi + wi_s + BAR_LABEL_Y_BACKBONE + LABEL_EM_BACKBONE
        joint = max(label_top_wo, label_top_wi) + BACKBONE_JOINT_PAD
        joint_wo = joint
        joint_wi = joint
        bracket_y = max(joint_wo, joint_wi) + BACKBONE_BRACKET_GAP
        x_left  = xi - width / 2
        x_right = xi + width / 2
        ax.plot([x_left, x_right], [bracket_y, bracket_y],
                color=bracket_color, linewidth=BRACKET_LINEWIDTH)
        ax.plot([x_left, x_left], [bracket_y, joint_wo],
                color=bracket_color, linewidth=BRACKET_LINEWIDTH)
        ax.plot([x_right, x_right], [bracket_y, joint_wi],
                color=bracket_color, linewidth=BRACKET_LINEWIDTH)
        ax.text(xi, bracket_y + BACKBONE_DELTA_TEXT_RISE, f"+{delta:.1f}",
                ha="center", va="bottom", fontsize=FS_BRACKET_DELTA,
                color=bracket_color, fontweight="bold")

    ax.set_title(
        title,
        fontsize=FS_PANEL_TITLE,
        pad=TITLE_PAD_TOP_PT,
        y=1.02,
        fontweight="bold",
    )
    # tick_params would override per-label font sizes, so set it first.
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK, pad=2)
    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_LABEL[b] for b in BACKBONES],
                       rotation=0, ha="center", fontsize=FS_XTICK_TOP, linespacing=1.0)
    ax.set_ylim(40, 108)
    ax.set_yticks(np.arange(40, 101, 10))
    ax.set_ylabel("Accuracy (%)", fontsize=FS_YLABEL, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------ Bottom row: PMv2 baselines --------------------- #

def _load_pmv2_baseline():
    rows = st.read_csv("personamemv2_baseline.csv")
    out = {ctx: {fmt: {} for fmt in ("MCQ", "Open")} for ctx in ("32k", "128k")}
    for r in rows:
        out[r["context"]][r["format"]][r["method"]] = (
            float(r["acc"]), float(r["std"]), int(r["is_ours"])
        )
    return out


def draw_pmv2_panel(ax, data_ctx, title, *, show_ylabel=True):
    """data_ctx: dict[fmt -> dict[method -> (acc, std, is_ours)]]."""
    x = np.arange(len(PMV2_METHODS))
    width = 0.38

    mcq_vals  = [data_ctx["MCQ"][m][0]  for m in PMV2_METHODS]
    mcq_stds  = [data_ctx["MCQ"][m][1]  for m in PMV2_METHODS]
    open_vals = [data_ctx["Open"][m][0] for m in PMV2_METHODS]
    open_stds = [data_ctx["Open"][m][1] for m in PMV2_METHODS]

    # Monochromatic Scientific-Blue gradient for cohesion with the top row:
    #   pale  -> baseline MCQ      light -> baseline Open
    #   base  -> PreCoMem MCQ      dark  -> PreCoMem Open
    base_mcq_color   = st.SCI_BLUE_PALE
    base_open_color  = st.SCI_BLUE_MID
    ours_mcq_color   = st.SCI_BLUE_BASE
    ours_open_color  = st.SCI_BLUE_DARK

    mcq_colors = [
        ours_mcq_color if data_ctx["MCQ"][m][2] else base_mcq_color
        for m in PMV2_METHODS
    ]
    open_colors = [
        ours_open_color if data_ctx["Open"][m][2] else base_open_color
        for m in PMV2_METHODS
    ]

    err_kwargs = dict(capsize=2.0, elinewidth=0.8, ecolor="#333")
    # MCQ: solid fill. Open-Ended: diagonal hatching for the second visual channel.
    # We need to set hatch color via rcParams temporarily to keep it crisp.
    prev_hatch_color = plt.rcParams.get("hatch.color", "black")
    prev_hatch_lw = plt.rcParams.get("hatch.linewidth", 1.0)
    plt.rcParams["hatch.color"] = "#ffffff"
    plt.rcParams["hatch.linewidth"] = 0.7

    ax.bar(x - width / 2, mcq_vals, width, yerr=mcq_stds,
           color=mcq_colors, edgecolor="#333", linewidth=0.6,
           error_kw=err_kwargs)
    ax.bar(x + width / 2, open_vals, width, yerr=open_stds,
           color=open_colors, edgecolor="#333", linewidth=0.6,
           hatch="//", error_kw=err_kwargs)

    plt.rcParams["hatch.color"] = prev_hatch_color
    plt.rcParams["hatch.linewidth"] = prev_hatch_lw

    # Numeric labels, nudged outward from the group centre so that the two
    # labels of a group never touch when the bars have similar heights.
    lbl_dx = 0.07
    for xi, (mv, ms, ov, os_) in enumerate(zip(mcq_vals, mcq_stds, open_vals, open_stds)):
        is_ours = data_ctx["MCQ"][PMV2_METHODS[xi]][2]
        c_mcq  = ours_mcq_color  if is_ours else "#445"
        c_open = ours_open_color if is_ours else "#234"
        fw = "bold" if is_ours else "normal"
        ax.text(xi - width / 2 - lbl_dx, mv + ms + BAR_LABEL_Y_PMV2, f"{mv:.1f}",
                ha="center", va="bottom", fontsize=FS_BAR_VALUE,
                color=c_mcq, fontweight=fw)
        ax.text(xi + width / 2 + lbl_dx, ov + os_ + BAR_LABEL_Y_PMV2, f"{ov:.1f}",
                ha="center", va="bottom", fontsize=FS_BAR_VALUE,
                color=c_open, fontweight=fw)

    # Highlight the gap between PreCoMem (Open) and the strongest baseline (Open).
    ours_idx = PMV2_METHODS.index("PreCoMem")
    baseline_open = [open_vals[i] for i in range(len(PMV2_METHODS)) if i != ours_idx]
    best_b_idx = int(np.argmax(baseline_open))
    # Map back to full index space.
    if best_b_idx >= ours_idx:
        best_b_idx += 1
    delta_open = open_vals[ours_idx] - open_vals[best_b_idx]
    ylim_top = 80.0
    if delta_open > 0:
        # Arms meet the value-label centerline for compact style parity with top row.
        label_base_l = open_vals[best_b_idx] + open_stds[best_b_idx] + BAR_LABEL_Y_PMV2
        label_base_r = open_vals[ours_idx] + open_stds[ours_idx] + BAR_LABEL_Y_PMV2
        joint_l = label_base_l + PMV2_LABEL_EM_CENTER + PMV2_BRACKET_TOUCH_OFFSET
        joint_r = label_base_r + PMV2_LABEL_EM_CENTER + PMV2_BRACKET_TOUCH_OFFSET
        y_top = max(joint_l, joint_r) + PMV2_BRACKET_ARM_RISE
        x_l = best_b_idx + width / 2
        x_r = ours_idx + width / 2
        ax.plot([x_l, x_r], [y_top, y_top],
                color=st.ACCENT_RED, linewidth=BRACKET_LINEWIDTH)
        ax.plot([x_l, x_l], [y_top, joint_l],
                color=st.ACCENT_RED, linewidth=BRACKET_LINEWIDTH)
        ax.plot([x_r, x_r], [y_top, joint_r],
                color=st.ACCENT_RED, linewidth=BRACKET_LINEWIDTH)
        ax.text((x_l + x_r) / 2, y_top + PMV2_DELTA_TEXT_OFFSET, f"+{delta_open:.2f}",
                ha="center", va="bottom", fontsize=FS_BRACKET_DELTA,
                color=st.ACCENT_RED, fontweight="bold")
        # Compact headroom so lower-row panels do not waste vertical space.
        ylim_top = max(ylim_top, y_top + PMV2_DELTA_TEXT_OFFSET + PMV2_YLIM_HEADROOM)

    ax.tick_params(axis="both", which="major", labelsize=FS_TICK, pad=2)
    ax.set_xticks(x)
    ax.set_xticklabels([PMV2_LABEL[m] for m in PMV2_METHODS],
                       rotation=0, ha="center", fontsize=FS_XTICK_BOTTOM)
    ax.set_ylim(50, ylim_top)
    ax.set_yticks(np.arange(50, 81, 10))
    if show_ylabel:
        ax.set_ylabel("Accuracy (%)", fontsize=FS_YLABEL, fontweight="bold")
    ax.set_title(title, fontsize=FS_PANEL_TITLE, pad=TITLE_PAD_BOTTOM_PT, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------- main --------------------------------- #

def main():
    # Base rcParams; panel/bar/legend sizes scale with `_UP` via FS_BASE_RC / FS_*.
    st.apply_paper_style(font_size=FS_BASE_RC)
    plt.rcParams["axes.labelweight"] = "bold"
    # White diagonal hatch on the Open-Ended bars in the bottom row keeps the
    # texture readable against the deep-blue fills without adding visual noise.
    plt.rcParams["hatch.color"] = "#ffffff"
    plt.rcParams["hatch.linewidth"] = 0.7

    backbone_data = _load_backbone_data()
    pmv2_data = _load_pmv2_baseline()

    # Shorter figure height + tighter inter-row gap to limit vertical paper footprint.
    fig = plt.figure(figsize=(15.8, 7.3))
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1.0, 1.02], hspace=0.55, figure=fig
    )

    # ---- Top: 4 backbone panels ----
    top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0],
                                           wspace=0.30)
    for col, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(top[col])
        draw_backbone_panel(ax, backbone_data[dataset], TOP_TITLES[col])
        if col != 0:
            ax.set_ylabel("")

    # ---- Bottom: 2 wide PMv2 baseline-comparison panels ----
    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                              wspace=0.18)
    ax_e = fig.add_subplot(bottom[0])
    ax_f = fig.add_subplot(bottom[1])
    draw_pmv2_panel(
        ax_e, pmv2_data["32k"],
        "(e) PersonaMem-v2 (32k): Memory-System Comparison",
        show_ylabel=True,
    )
    draw_pmv2_panel(
        ax_f, pmv2_data["128k"],
        "(f) PersonaMem-v2 (128k): Memory-System Comparison",
        show_ylabel=False,
    )

    # ---- Two legends, separated by row ----
    top_handles = [
        plt.Rectangle((0, 0), 1, 1, color=st.SCI_BLUE_MID,
                      label="Answer model + its native memory (O-Mem)", ec="#333", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, color=st.SCI_BLUE_DARK,
                      label="Same answer model + PreCoMem on a smaller memory backbone", ec="#333", lw=0.5,
                      hatch="//"),
        plt.Line2D([0], [0], color=st.ACCENT_RED, lw=1.4,
                   label=r"Absolute gain $\Delta$"),
    ]
    leg_top = fig.legend(
        handles=top_handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.03), ncol=3,
        frameon=True, fancybox=False,
        facecolor=LEGEND_FRAME_FACE, edgecolor=LEGEND_FRAME_EDGE,
        framealpha=LEGEND_FRAME_ALPHA,
        fontsize=FS_FIG_LEGEND_ROW1, labelspacing=0.2, handlelength=1.2,
        handletextpad=0.45, columnspacing=1.05, borderaxespad=0.2,
        borderpad=0.45,
    )
    _style_fig_legend_dashed_frame(leg_top)

    bottom_handles = [
        plt.Rectangle((0, 0), 1, 1, color=st.SCI_BLUE_PALE,
                      label="Baseline (MCQ)", ec="#333", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, color=st.SCI_BLUE_MID,
                      label="Baseline (Open-Ended)", ec="#333", lw=0.5,
                      hatch="//"),
        plt.Rectangle((0, 0), 1, 1, color=st.SCI_BLUE_BASE,
                      label="PreCoMem (MCQ)", ec="#333", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, color=st.SCI_BLUE_DARK,
                      label="PreCoMem (Open-Ended)", ec="#333", lw=0.5,
                      hatch="//"),
    ]
    leg_mid = fig.legend(
        handles=bottom_handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.515), ncol=4,
        frameon=True, fancybox=False,
        facecolor=LEGEND_FRAME_FACE, edgecolor=LEGEND_FRAME_EDGE,
        framealpha=LEGEND_FRAME_ALPHA,
        fontsize=FS_FIG_LEGEND_ROW2, labelspacing=0.2, handlelength=1.2,
        handletextpad=0.45, columnspacing=0.95, borderaxespad=0.2,
        borderpad=0.45,
    )
    _style_fig_legend_dashed_frame(leg_mid)

    # Tight vertical margins; slightly wider left for y-axis labels.
    fig.subplots_adjust(left=0.055, right=0.995, top=0.885, bottom=0.07)

    st.save(fig, "locomo_backbone_overview")
    plt.close(fig)


if __name__ == "__main__":
    main()
