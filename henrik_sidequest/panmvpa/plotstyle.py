"""One shared matplotlib style for every figure in the project -- see docs/plot_style.md.

Import it, call ``apply()`` once, build the figure with the helpers, and ``save(fig, stem)``
(writes both PDF for the paper and PNG for viewing). Nothing is set per-figure; change the look
here and every plot follows.

    from panmvpa import plotstyle as ps
    ps.apply()
    fig, ax = ps.plt.subplots(figsize=(ps.HALF, 2.8))
    ...
    ps.style_ax(ax); ps.legend(ax)
    ps.titles(fig, "the claim", "n=10 | 1-45 min | seed 0")
    ps.save(fig, out_dir / "curves")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- dimensions ------------------------------------------------------------
TEXTWIDTH = 6.5                 # inches, the document \textwidth
FULL = TEXTWIDTH                # multi-panel figures
HALF = TEXTWIDTH * 0.55         # single-panel figures

# --- fonts (never hardcode a size; reference FS) ---------------------------
FS = {"title": 9, "label": 8, "tick": 7, "subtitle": 7.5, "legend": 6.5, "annot": 5.5}

# --- palette ---------------------------------------------------------------
SUNSET = LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20",
    "#E09040", "#EDBC70", "#F5DCA8",
])
# Diverging variant for SIGNED data (e.g. correlation matrices with anticorrelation): the warm
# sunset on one side, its cool complement on the other, white at zero. Use with symmetric limits
# (vmin=-vmax) so white lands on 0. Sequential SUNSET everywhere the data is one-signed.
SUNSET_DIV = LinearSegmentedColormap.from_list("sunset_div", [
    "#123A44", "#2C7A8C", "#5FA8B5", "#A9D3DA",   # cool complement (negative)
    "#FFFFFF",                                      # zero
    "#F1C27A", "#E0903F", "#CC5A20", "#7A1F0C",    # warm sunset (positive)
])
ACCENT = "#CC5A20"              # mid-orange, the single accent
GREY = "#999999"               # de-emphasised series

INK, SUBINK, PANEL = "#111111", "#555555", "#222222"
SPINE, TICKINK = "#cccccc", "#333333"

SAVE_KW = dict(bbox_inches="tight", pad_inches=0.01, facecolor="white")


def apply() -> None:
    """Set the global rcParams: serif, sizes from FS, warm/soft axes."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "axes.edgecolor": SPINE, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.color": TICKINK, "ytick.color": TICKINK,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.2,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def sunset_colors(n: int) -> list:
    """n line colours sampled from SUNSET, avoiding the near-black/near-white extremes."""
    if n <= 1:
        return [ACCENT]
    return [SUNSET(0.82 - 0.65 * i / (n - 1)) for i in range(n)]


def style_ax(ax) -> None:
    """Top/right spines off; left/bottom light grey; inward dark-grey ticks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(direction="in", color=TICKINK, which="both")


def titles(fig, claim: str, metadata: str | None = None, top: float = 0.84) -> None:
    """Bold claim (suptitle) with a grey metadata line below it; reserve headroom.

    Lay the axes out first (into the lower ``top`` fraction), then hang the title block in the
    band above -- claim near the top, metadata a clear gap below -- so they never collide.
    """
    fig.tight_layout(rect=[0, 0, 1, top])
    band = 1.0 - top
    fig.suptitle(claim, fontsize=FS["title"], fontweight="bold", color=INK,
                 y=top + band * 0.82, va="top")
    if metadata:
        fig.text(0.5, top + band * 0.34, metadata, ha="center", va="top",
                 fontsize=FS["subtitle"], color=SUBINK)


def panel(ax, i: int, name: str) -> None:
    """Panel label merged into the axes title: '(a)  name'."""
    ax.set_title(f"({chr(97 + i)})  {name}", fontsize=FS["title"] - 1,
                 fontweight="semibold", color=PANEL, pad=5)


def legend(ax, **kw):
    return ax.legend(frameon=False, fontsize=FS["legend"], labelcolor=TICKINK, **kw)


def colorbar(fig, im, ax, label: str | None = None):
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03, aspect=25)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=FS["tick"])
    if label:
        cb.set_label(label, fontsize=FS["label"])
    return cb


def save(fig, stem) -> str:
    """Write <dir>/pdf/<name>.pdf (paper) and <dir>/png/<name>.png (viewing) from a stem like
    <dir>/<name>. Each format lands in its own sibling folder. Returns the .png path."""
    stem = Path(stem)
    out = {}
    for ext in ("pdf", "png"):
        d = stem.parent / ext
        d.mkdir(parents=True, exist_ok=True)
        out[ext] = d / f"{stem.name}.{ext}"
        fig.savefig(out[ext], **SAVE_KW)
    plt.close(fig)
    return str(out["png"])
