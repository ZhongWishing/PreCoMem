"""One-row composite: frontier + memory scalability (three panels).

This script keeps the original visual styling from:
  - plot_locomo_frontier.py
  - plot_memory_scalability.py
and only provides a shared container to place them in one row.

No data values are changed.
"""

from __future__ import annotations

import _style as st
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Typography: same *relative* hierarchy as 2_plot_ablation_dimobench_row.py, but
# scaled up for this three-panel row (axis/title text was still too small in LaTeX).
_UP = 1.9
# Point sizes below are tuned for the 17.5 in canvas set in main(); at the paper
# width (5.5 in) they render at roughly 0.31x, i.e. ticks ~5.4 pt, axis labels ~6 pt.
FS_BASE = int(round(15 * _UP))  # 20
FS_TITLE = 19
FS_AXIS = 21
FS_TICK = 18
FS_LEGEND = 17
FS_LEGEND_MEM = FS_LEGEND
# (b)(c) drift callouts.
FS_DRIFT = 19

# Panel (a) in-plot text scales (independent): baseline method labels + K markers.
# In-plot method names / K markers keep the submission-version size relative to the
# panel (18.2 pt on the old 28.5 in canvas == 11.2 pt here); larger text overlaps
# the clustered EverMemOS / MemOS / TiMem / O-Mem labels.
FRONTIER_INPLOT_TEXT_SCALE = 0.93
FRONTIER_K_TEXT_EXTRA_SCALE = 1.25

# Panel (a) lower-right "PreCoMem (Ours)" line legend only.
# Keep slightly smaller than (b)(c), but larger than before and closer to corner.
FRONTIER_PRCOMEM_LEGEND_SCALE = 0.9

# Panel (a) viewing window only: tighter limits remove inner whitespace and
# enlarge the core curve/node region without changing any plotted data/links.
FRONTIER_CORE_XLIM = (450, 6100)
FRONTIER_CORE_YLIM = (44, 92)

# Same fraction for (b) and (c): drift labels share one baseline height across panels.
# High enough to stay clear of the bottom-right legend on (c); tune 0.01 steps if needed.
DRIFT_LABEL_Y_FRAC = 0.3

# Marker shape per baseline (same as plot_locomo_frontier.py).
BASELINE_MARKER = {
    "O-Mem":     ("v", st.SCI_BLUE_BASE),
    "MemBox":    ("D", st.ACCENT_TEAL),
    "GAM":       (">", st.ACCENT_GRAY),
    "MemoryOS":  ("s", st.ACCENT_AMBER),
    "TiMem":     ("h", st.ACCENT_PURPLE),
    "MemOS":     ("p", st.ACCENT_GREEN),
    "EverMemOS": ("X", st.SCI_BLUE_MID),
    "Mem0":      ("^", st.ACCENT_RED),
    "A-Mem":     ("<", "#999999"),
}

LABEL_POS = {
    "EverMemOS": (1480, 76.6),
    "MemOS":     (2360, 74.6),
    "TiMem":     (3720, 81.8),
    "O-Mem":     (3720, 77.7),
}

DIRECT_LABEL_STYLE = {
    "Mem0":     (-250, -2.9, "left", "center"),
    "A-Mem":    (-130, 0.0, "right", "center"),
    "MemoryOS": (0, -2.2, "center", "top"),
    "GAM":      (0, -2.2, "center", "top"),
    "MemBox":   (95, 2.1, "left", "center"),
}

LEADER_CURVE = {
    "EverMemOS": "arc3,rad=0.10",
    "MemOS": "arc3,rad=0.06",
    "TiMem": "arc3,rad=0.12",
    "O-Mem": "arc3,rad=-0.05",
}


def _draw_frontier(ax: plt.Axes) -> None:
    fs_k = 13 * FRONTIER_INPLOT_TEXT_SCALE * FRONTIER_K_TEXT_EXTRA_SCALE
    fs_baseline = 12 * FRONTIER_INPLOT_TEXT_SCALE
    leg_s = FRONTIER_PRCOMEM_LEGEND_SCALE
    pcm_legend_fs = FS_LEGEND_MEM * leg_s
    legend_marker_pt = 9 * (pcm_legend_fs / 12.0)

    rows = st.read_csv("pareto_frontier_locomo.csv")

    pcm_curve = [r for r in rows if r["is_precomem_curve"] == "1"]
    pcm_curve.sort(key=lambda r: int(r["k"]))
    pcm_tokens = np.array([float(r["token_used"]) for r in pcm_curve])
    pcm_acc = np.array([float(r["accuracy_overall"]) for r in pcm_curve])
    pcm_k = np.array([int(r["k"]) for r in pcm_curve])

    baselines = [r for r in rows if r["is_precomem_curve"] == "0"]

    pcm_color = st.SCI_BLUE_DARK
    ax.plot(pcm_tokens, pcm_acc, color=pcm_color, linewidth=3.0, zorder=2,
            label="PreCoMem (Ours)")
    ax.scatter(pcm_tokens, pcm_acc, color=pcm_color, s=55, zorder=3)

    labeled = {1, 10}
    for kv, t, a in zip(pcm_k, pcm_tokens, pcm_acc):
        if int(kv) in labeled:
            ax.scatter(t, a, color=pcm_color, s=140, zorder=4,
                       edgecolors="white", linewidth=1.6)
            offy = -3.9 if int(kv) == 1 else 1.6
            # K=1 sits at the left edge of the window: anchor its label to the right.
            ha = "left" if int(kv) == 1 else "center"
            offx = 60 if int(kv) == 1 else 0
            ax.text(t + offx, a + offy, f"K={int(kv)}", ha=ha, va="bottom",
                    fontsize=fs_k, color=pcm_color, fontweight="bold")

    for b in baselines:
        name = b["method"]
        marker, color = BASELINE_MARKER[name]
        tok = float(b["token_used"])
        acc = float(b["accuracy_overall"])
        ax.scatter(tok, acc, marker=marker, c=color, s=180,
                   edgecolors="white", linewidth=1.4, zorder=4)
        if name in DIRECT_LABEL_STYLE:
            dx, dy, ha, va = DIRECT_LABEL_STYLE[name]
            ax.text(tok + dx, acc + dy, name,
                    ha=ha, va=va,
                    fontsize=fs_baseline, color="#222", fontweight="bold",
                    zorder=5)
            continue

        lx, ly = LABEL_POS[name]
        ax.annotate(
            name, xy=(tok, acc), xytext=(lx, ly),
            ha="center", va="center",
            fontsize=fs_baseline, color="#222", fontweight="bold",
            arrowprops=dict(
                arrowstyle="-",
                linestyle=(0, (3, 2)),
                color="#888",
                linewidth=0.9,
                shrinkA=4, shrinkB=8,
                connectionstyle=LEADER_CURVE[name],
            ),
            zorder=5,
        )

    ax.set_xlabel("Answer Token Usage", fontsize=FS_AXIS, fontweight="bold",
                  labelpad=6)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=FS_AXIS, fontweight="bold",
                  labelpad=6)
    ax.set_xlim(*FRONTIER_CORE_XLIM)
    ax.set_ylim(*FRONTIER_CORE_YLIM)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.4)
        sp.set_color("#333")

    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="major",
        width=1.4,
        length=6,
        labelsize=FS_TICK,
        pad=2,
    )

    legend_line = plt.Line2D(
        [0],
        [0],
        color=pcm_color,
        lw=2.5,
        marker="o",
        markersize=legend_marker_pt,
        label="PreCoMem (Ours)",
    )
    ax.legend(
        handles=[legend_line],
        loc="lower right",
        bbox_to_anchor=(0.995, 0.14),  # raised so it clears the A-Mem marker
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=pcm_legend_fs,
        handlelength=1.8 * leg_s,
        handletextpad=0.5 * leg_s,
        borderpad=0.4 * leg_s,
        labelspacing=0.4 * leg_s,
    )

    ax.set_title(
        "(a) Accuracy\u2013Efficiency Frontier",
        fontsize=FS_TITLE,
        pad=6.0,
        fontweight="bold",
    )


def _draw_memory_panel(
    ax_left,
    csv_name,
    drift_points,
    title,
    xticks,
    xlim,
):
    rows = st.read_csv(csv_name)
    turns = np.array([int(r["turn"]) for r in rows])
    n_raw = np.array([float(r["N_raw"]) for r in rows])
    n_eff = np.array([float(r["N_eff"]) for r in rows])

    color_raw = st.SCI_BLUE_BASE
    color_eff = st.ACCENT_AMBER

    line_raw, = ax_left.plot(turns, n_raw, color=color_raw, marker="o",
                             markersize=4.5, linewidth=2.4, label="$N_\\mathrm{raw}$")
    ax_left.set_xlabel("Dialogue Turn", fontsize=FS_AXIS, fontweight="bold",
                       labelpad=6)
    ax_left.set_ylabel("$N_\\mathrm{raw}$ (raw items)", fontsize=FS_AXIS,
                       fontweight="bold", color=color_raw)
    ax_left.tick_params(
        axis="y", labelcolor=color_raw, labelsize=FS_TICK, pad=2
    )
    ax_left.tick_params(axis="x", labelsize=FS_TICK, pad=2)

    ax_right = ax_left.twinx()
    line_eff, = ax_right.plot(turns, n_eff, color=color_eff, marker="s",
                              markersize=4.5, linewidth=2.4, linestyle="--",
                              label="$N_\\mathrm{eff}$")
    ax_right.set_ylabel("$N_\\mathrm{eff}=\\sum_m w_t(m)$", fontsize=FS_AXIS,
                        fontweight="bold", color=color_eff)
    ax_right.tick_params(
        axis="y", labelcolor=color_eff, labelsize=FS_TICK, pad=2
    )

    for dp in drift_points:
        ax_left.axvline(x=dp, color=st.ACCENT_RED, linestyle=":",
                        linewidth=1.2, alpha=0.75)
        ymin, ymax = ax_left.get_ylim()
        y_txt = ymin + (ymax - ymin) * DRIFT_LABEL_Y_FRAC
        ax_left.text(
            dp + 4,
            y_txt,
            f"drift\n@{dp}",
            fontsize=FS_DRIFT,
            color=st.ACCENT_RED,
            va="bottom",
            fontweight="bold",
        )

    ax_left.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=6)
    ax_left.set_xticks(xticks)
    ax_left.set_xlim(*xlim)
    ax_left.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)

    ax_left.legend(
        handles=[line_raw, line_eff],
        loc="lower right",
        fontsize=FS_LEGEND_MEM,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.6,
        handletextpad=0.6,
        borderpad=0.5,
        labelspacing=0.4,
    )

    ax_right.axhspan(25.8, 28.5, color=color_eff, alpha=0.10, zorder=0)


def main() -> None:
    st.apply_paper_style(font_size=FS_BASE)
    plt.rcParams["axes.labelweight"] = "bold"

    # Layout: extra horizontal gap only between (b) and (c) so the right y-axis
    # label on (b) (twin) does not overlap the left y-axis label on (c).
    # Canvas shrunk from 28.5 x 7.4 in (same aspect ratio) so that the point-sized
    # text renders ~1.8x larger at the paper width; data/layout are unchanged.
    fig = plt.figure(figsize=(17.5, 4.8))
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.22, 1.28, 1.28],
        wspace=0.50,
    )
    axes = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]))

    _draw_frontier(axes[0])

    _draw_memory_panel(
        axes[1],
        csv_name="memory_scalability_dimo350.csv",
        drift_points=[200],
        title="(b) DiMoBench (350 turns, 1 drift)",
        xticks=np.arange(0, 360, 50),
        xlim=(-5, 355),
    )

    _draw_memory_panel(
        axes[2],
        csv_name="memory_scalability_dimo_xl.csv",
        drift_points=[280, 580, 870],
        title="(c) DiMoBench-XL (1050 turns, 3 drifts)",
        xticks=np.arange(0, 1100, 200),
        xlim=(-10, 1060),
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.17)

    st.save(fig, "frontier_memory_scalability_row")
    plt.close(fig)


if __name__ == "__main__":
    main()
