"""One-row composite for three ablation figures used in the paper.

This script only adjusts visual styling and layout. It does not change any data.
Inputs:
  - component_ablation.csv
  - dimobench_dynamics.csv
  - dimobench_operator_ablation.csv
Output:
  - ablation_dimobench_row.pdf/.png
"""

from __future__ import annotations

import _style as st
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Typography / primitives: same pattern as 3_plot_frontier_memory_row.py — tune `_UP`
# to scale text and (lightly) strokes/markers together for LaTeX placement.
_UP = 1.6

FS_BASE = int(round(15 * _UP))
FS_TITLE = int(round(14 * _UP))
FS_AXIS = int(round(15 * _UP))
FS_TICK = int(round(13 * _UP))
FS_LEGEND = int(round(12 * _UP))
FS_VALUE = int(round(12 * _UP))
FS_ANNOT = int(round(11 * _UP))
FS_BENCH = int(round(13 * _UP))

# Panel (b): legend slightly smaller than FS_LEGEND; mirrors old max(FS_LEGEND-1.5, 9.5).
FS_LEGEND_DYNAMICS = max(
    FS_LEGEND - max(1, int(round(1.5 * _UP))),
    int(round(9.5 * _UP)),
)
FS_LEGEND_OPERATOR = FS_LEGEND_DYNAMICS
# Axes-fraction y anchor of the legends placed under the x-axis label of (b)/(c).
LEGEND_BELOW_Y = -0.27

LABELPAD_AXIS = max(4, int(round(6 * _UP)))
TITLEPAD_BENCH = max(4, int(round(5 * _UP)))

# Cap growth so very large _UP does not yield unreadably thick lines (matches intent of frontier row).
_V = min(_UP, 1.35)
LW_MAIN = 2.6 * _V
LW_ALT = 2.1 * _V
LW_GUIDE = 1.0 * _V
LW_VLINE_STRONG = 1.4 * _V
LW_VLINE_SOFT = 1.2 * _V
MS_LINE = int(round(6 * _V))
LEGEND_HANDLELEN = 2.4 * min(_UP, 1.25)
BAR_EDGE = max(0.45, 0.65 * min(_UP, 1.25))

VARIANT_ORDER = ["full", "no_rr", "no_weight", "no_surp"]
VARIANT_LABEL = {
    "full": "Full",
    "no_rr": "w/o RR",
    "no_weight": "w/o CW",
    "no_surp": "w/o ES",
}
VARIANT_COLORS = [
    st.SCI_BLUE_DARK,
    st.SCI_BLUE_BASE,
    st.SCI_BLUE_MID,
    st.SCI_BLUE_LIGHT,
]

# CSV keys stay as-is; the paper refers to the 500-question LongMemEval-S split.
BENCH_LABEL = {"LoCoMo": "LoCoMo", "LongMemEval": "LongMemEval-S"}

# Figure-fraction gap kept between the panel title "(a) ..." and the first
# benchmark sub-title; the two bar axes are compressed towards the bottom.
BAR_COLUMN_TOP_PAD = 0.05

# Match standalone line plots (e.g. plot_dimobench_operator_ablation.py: figsize 9 x 6.5).
_LINE_BOX_ASPECT = 6.5 / 9.0


def _fig_title_centered_over_cell(
    fig: plt.Figure,
    subplot_spec: gridspec.SubplotSpec,
    title: str,
    *,
    fontsize: float,
    y_fig: float,
) -> None:
    """Place a panel title centered above the given GridSpec cell (figure coords)."""
    pos = subplot_spec.get_position(fig)
    fig.text(
        pos.x0 + pos.width / 2,
        y_fig,
        title,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        transform=fig.transFigure,
    )


def _draw_component_ablation(fig: plt.Figure, cell):
    rows = st.read_csv("component_ablation.csv")
    grouped = {b: {} for b in ["LoCoMo", "LongMemEval"]}
    for r in rows:
        grouped[r["benchmark"]][r["variant"]] = float(r["accuracy"])

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=cell, hspace=0.45
    )
    axes = []
    for idx, bench in enumerate(["LoCoMo", "LongMemEval"]):
        ax = fig.add_subplot(inner[idx])
        axes.append(ax)
        values = [grouped[bench][v] for v in VARIANT_ORDER]
        full_value = grouped[bench]["full"]
        y = np.arange(len(VARIANT_ORDER))

        ax.barh(
            y,
            values,
            color=VARIANT_COLORS,
            height=0.76,
            edgecolor="black",
            linewidth=BAR_EDGE,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [VARIANT_LABEL[v] for v in VARIANT_ORDER], ha="right", fontsize=FS_TICK
        )
        ax.invert_yaxis()
        ax.set_title(
            BENCH_LABEL[bench], fontsize=FS_BENCH, pad=TITLEPAD_BENCH, fontweight="bold"
        )
        ax.set_xlim(0, 100)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        if idx == 0:
            # Both sub-panels share the 0-100 scale; the upper tick labels would
            # otherwise run into the lower panel's benchmark title.
            ax.tick_params(axis="x", labelbottom=False)

        # Value and delta are placed with point offsets so the delta always
        # starts just after the value text whatever the font size (Times digits
        # are about 0.5 em wide).
        gap_pt = 0.3 * FS_VALUE
        for i, val in enumerate(values):
            val_txt = f"{val:.2f}"
            ax.annotate(
                val_txt,
                xy=(val, i),
                xytext=(gap_pt, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=FS_VALUE,
                fontweight="bold",
            )
            if VARIANT_ORDER[i] != "full":
                delta = val - full_value
                ax.annotate(
                    f"({delta:+.2f})",
                    xy=(val, i),
                    xytext=(gap_pt + 0.5 * FS_VALUE * len(val_txt) + 0.35 * FS_VALUE, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=FS_VALUE,
                    fontweight="bold",
                    color=st.ACCENT_RED,
                )

        ax.grid(axis="x", linestyle="--", alpha=0.30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 1:
            ax.set_xlabel(
                "Accuracy (%)",
                fontsize=FS_AXIS,
                fontweight="bold",
                labelpad=LABELPAD_AXIS,
            )

    return axes


def _compress_bar_column(axes) -> None:
    """Shift the two bar axes down so the first benchmark sub-title clears the
    panel title. Must run after fig.subplots_adjust(), which re-lays out every
    gridspec-managed axes."""
    bottom = min(ax.get_position().y0 for ax in axes)
    top = max(ax.get_position().y1 for ax in axes)
    scale = (top - BAR_COLUMN_TOP_PAD - bottom) / (top - bottom)
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([
            pos.x0,
            bottom + (pos.y0 - bottom) * scale,
            pos.width,
            pos.height * scale,
        ])


def _draw_dimobench_dynamics(ax: plt.Axes) -> None:
    rows = st.read_csv("dimobench_dynamics.csv")
    turns = np.array([int(r["turn"]) for r in rows])
    series_style = {
        "PreCoMem": dict(
            color=st.SCI_BLUE_DARK,
            marker="o",
            linestyle="-",
            linewidth=LW_MAIN,
            markersize=MS_LINE,
            zorder=10,
        ),
        "EverMemOS": dict(
            color=st.SCI_BLUE_BASE,
            marker="X",
            linestyle="--",
            linewidth=LW_ALT,
            markersize=MS_LINE,
        ),
        "TiMem": dict(
            color=st.ACCENT_PURPLE,
            marker="h",
            linestyle="-.",
            linewidth=LW_ALT,
            markersize=MS_LINE,
        ),
        "O-Mem": dict(
            color=st.ACCENT_AMBER,
            marker="s",
            linestyle="--",
            linewidth=LW_ALT,
            markersize=MS_LINE,
        ),
        "Mem0": dict(
            color=st.ACCENT_RED,
            marker="^",
            linestyle=":",
            linewidth=LW_ALT,
            markersize=MS_LINE,
        ),
    }

    for name, style in series_style.items():
        y = np.array([float(r[name]) for r in rows])
        ax.plot(turns, y, label=name, **style)

    ax.axhline(
        y=84,
        color=st.SCI_BLUE_DARK,
        linestyle="--",
        linewidth=LW_GUIDE,
        alpha=0.55,
    )
    ax.text(
        352,
        84.8,
        "PreCoMem stabilizes at 84%",
        fontsize=FS_ANNOT,
        color=st.SCI_BLUE_DARK,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    ax.axvline(
        200,
        color=st.ACCENT_RED,
        linestyle=":",
        linewidth=LW_VLINE_STRONG,
        alpha=0.85,
    )
    # Top-left corner, right of the drift line, above the 84% guide and its caption.
    ax.text(
        203,
        100.0,
        "Belief Shift (T=200)",
        fontsize=FS_ANNOT,
        color=st.ACCENT_RED,
        va="top",
        ha="left",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Dialogue Turns",
        fontsize=FS_AXIS,
        fontweight="bold",
        labelpad=LABELPAD_AXIS,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=FS_AXIS, fontweight="bold")
    ax.set_xticks(np.arange(150, 360, 40))
    ax.set_xlim(145, 355)
    ax.set_ylim(20, 101)
    ax.set_yticks(np.arange(20, 91, 10))
    ax.tick_params(axis="both", labelsize=FS_TICK)
    # Legend below the axes: at the larger font an in-axes legend would cover
    # the post-drift curves.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_BELOW_Y),
        ncol=3,
        fontsize=FS_LEGEND_DYNAMICS,
        frameon=False,
        columnspacing=1.0,
        labelspacing=0.25,
        handlelength=LEGEND_HANDLELEN,
        handletextpad=0.45,
        markerscale=0.85,
    )
    ax.set_box_aspect(_LINE_BOX_ASPECT)
    ax.set_anchor("S")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_dimobench_operator(ax: plt.Axes) -> None:
    rows = st.read_csv("dimobench_operator_ablation.csv")
    turns = np.array([int(r["turn"]) for r in rows])
    series_style = {
        "full_PCM": dict(
            label="Full PreCoMem",
            color=st.SCI_BLUE_DARK,
            marker="o",
            linestyle="-",
            linewidth=LW_MAIN,
            markersize=MS_LINE,
            zorder=10,
        ),
        "no_PROFILE": dict(
            label="w/o PROFILE",
            color=st.ACCENT_GREEN,
            marker="s",
            linestyle="--",
            linewidth=LW_ALT,
            markersize=MS_LINE,
        ),
        "no_CORRECT": dict(
            label="w/o CORRECT",
            color=st.ACCENT_PURPLE,
            marker="d",
            linestyle="-.",
            linewidth=LW_ALT,
            markersize=MS_LINE,
        ),
        "no_MAINTAIN": dict(
            label="w/o MAINTAIN",
            color=st.ACCENT_AMBER,
            marker="^",
            linestyle=":",
            linewidth=LW_ALT,
            markersize=MS_LINE,
        ),
    }

    for name, style in series_style.items():
        y = np.array([float(r[name]) for r in rows])
        ax.plot(turns, y, **style)

    ax.axvline(
        150, color="black", linestyle="-", linewidth=LW_VLINE_SOFT, alpha=0.30
    )
    # One line along the bottom edge: the two-line version ran into the
    # w/o CORRECT trough once the annotation font was enlarged.
    ax.text(
        152,
        21.5,
        "Intervention Start (T=150)",
        fontsize=FS_ANNOT,
        fontweight="bold",
        color="black",
        va="bottom",
    )
    ax.axvline(
        200,
        color=st.ACCENT_RED,
        linestyle=":",
        linewidth=LW_VLINE_STRONG,
        alpha=0.85,
    )
    ax.text(
        202,
        90,
        "Belief Shift (T=200)",
        fontsize=FS_ANNOT,
        color=st.ACCENT_RED,
        va="top",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Dialogue Turns",
        fontsize=FS_AXIS,
        fontweight="bold",
        labelpad=LABELPAD_AXIS,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=FS_AXIS, fontweight="bold")
    ax.set_xticks(np.arange(150, 360, 40))
    ax.set_xlim(148, 355)
    ax.set_ylim(20, 95)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_BELOW_Y),
        ncol=2,
        fontsize=FS_LEGEND_OPERATOR,
        frameon=False,
        columnspacing=1.0,
        labelspacing=0.25,
        handlelength=LEGEND_HANDLELEN,
        handletextpad=0.45,
        markerscale=0.85,
    )
    ax.set_box_aspect(_LINE_BOX_ASPECT)
    ax.set_anchor("S")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    st.apply_paper_style(font_size=FS_BASE)
    plt.rcParams["axes.labelweight"] = "bold"

    # Slightly wider line columns so (b)(c) match standalone 9:6.5 box aspect without looking squeezed.
    fig = plt.figure(figsize=(17.2, 5.7))
    gs = gridspec.GridSpec(
        1,
        3,
        width_ratios=[1.08, 1.14, 1.14],
        wspace=0.26,
        figure=fig,
    )

    bar_axes = _draw_component_ablation(fig, gs[0])
    ax_dyn = fig.add_subplot(gs[1])
    _draw_dimobench_dynamics(ax_dyn)
    ax_op = fig.add_subplot(gs[2])
    _draw_dimobench_operator(ax_op)

    fig.subplots_adjust(left=0.045, right=0.995, top=0.89, bottom=0.25)
    _compress_bar_column(bar_axes)
    title_y = 0.965
    _fig_title_centered_over_cell(
        fig, gs[0], "(a) Component Ablation", fontsize=FS_TITLE, y_fig=title_y
    )
    _fig_title_centered_over_cell(
        fig,
        gs[1],
        "(b) Adaptation Dynamics on DiMoBench",
        fontsize=FS_TITLE,
        y_fig=title_y,
    )
    _fig_title_centered_over_cell(
        fig,
        gs[2],
        "(c) Operator Contribution on DiMoBench",
        fontsize=FS_TITLE,
        y_fig=title_y,
    )

    st.save(fig, "ablation_dimobench_row")
    plt.close(fig)


if __name__ == "__main__":
    main()
