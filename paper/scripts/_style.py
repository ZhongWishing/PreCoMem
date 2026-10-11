"""Shared style helpers for the paper figures (Figures 3, 4 and 5).

All three scripts in this directory use the same "Scientific Blue" palette and
the same Times-based typography so the figures look consistent in the paper.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# Scientific Blue Theme — primary palette (matches the original ablation figure).
SCI_BLUE_DARK = "#08519c"
SCI_BLUE_BASE = "#2171b5"
SCI_BLUE_MID = "#6baed6"
SCI_BLUE_LIGHT = "#9ecae1"
SCI_BLUE_PALE = "#c6dbef"

# Accent / contrast hues used sparingly to encode non-PreCoMem methods or annotations.
ACCENT_RED = "#d62728"
ACCENT_AMBER = "#e08e00"
ACCENT_GREEN = "#1f6f4a"
ACCENT_PURPLE = "#6a51a3"
ACCENT_TEAL = "#3690c0"
ACCENT_GRAY = "#737373"

# Default ordering of operator-removal variants (used by ablation plots).
DEFAULT_VARIANT_ORDER = ["full", "no_rr", "no_weight", "no_surp"]
DEFAULT_VARIANT_LABEL = {
    "full": "Full PreCoMem",
    "no_rr": "w/o RR",
    "no_weight": "w/o CW",
    "no_surp": "w/o ES",
}
DEFAULT_VARIANT_COLOR = [
    SCI_BLUE_DARK,
    SCI_BLUE_BASE,
    SCI_BLUE_MID,
    SCI_BLUE_LIGHT,
]


def apply_paper_style(font_size: int = 19) -> None:
    """Apply the unified 'Times New Roman serif' style used across every figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.labelweight": "bold",
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": max(font_size - 5, 11),
        "ytick.labelsize": max(font_size - 5, 11),
        "legend.fontsize": max(font_size - 6, 11),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Project paths (resolved relative to this file): paper/scripts -> paper/data, paper/output.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"


def read_csv(name: str) -> List[Dict[str, str]]:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, basename: str, also_paper: bool = False) -> None:
    """Save the figure as PDF and PNG under paper/output/ (``also_paper`` is kept for API compatibility)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{basename}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{basename}.png", dpi=300, bbox_inches="tight")
    print(f"saved {basename}.pdf/.png -> {OUTPUT_DIR}")
