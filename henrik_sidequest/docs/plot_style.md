# Plot Style Guide

Paper setup: `\documentclass[12pt]{article}`, `\geometry{margin=1in}`, letter paper.
Notebook environment: Jupyter in VS Code, matplotlib, PDF output.

All of this is implemented once in `panmvpa/plotstyle.py`; every plotting function imports it
rather than setting anything per-figure.

---

## Dimensions

- **Figure width:** `TEXTWIDTH = 6.5` inches (= 469.8pt, the document's `\textwidth`).
- **Full-width figures** (multi-panel): `figsize=(TEXTWIDTH, height)`. Heights typically 2.8–3.2 in.
- **Half-width figures** (single-panel): `figsize=(TEXTWIDTH * 0.55, 2.8)`.
- **Aspect ratios** are set per-figure, not globally. The width is the anchor; height adjusts to content.

## LaTeX Integration

- Figures are generated in a Jupyter notebook, saved as PDF, then copied into the LaTeX project's `figures/` directory.
- LaTeX includes them with `\includegraphics[width=0.85\textwidth]{figures/filename}`.
- The 15% downscale is acceptable — font sizes stay legible.
- `USE_TEX = False` by default (DejaVu Serif fallback). Set `True` after installing `cm-super` (`tlmgr install cm-super type1ec`) for exact Computer Modern font matching.

## Output Format

- **PDF for the paper, PNG for viewing on the hub.** Both are written for every figure. PDF is
  vector — text stays sharp at any zoom, smaller file size.
- Save with: `bbox_inches="tight", pad_inches=0.01, facecolor="white"`.
- These are collected in a `SAVE_KW` dict and unpacked at every `savefig` call.

## Fonts

- **Family:** Serif. DejaVu Serif as fallback, Computer Modern Roman when `USE_TEX = True`.
- **Sizes** (defined once in an `FS` dict, referenced everywhere):

  | Key        | Size (pt) | Used for                          |
  |------------|-----------|-----------------------------------|
  | `title`    | 9         | `suptitle`                        |
  | `label`    | 8         | Axis labels, colorbar labels      |
  | `tick`     | 7         | Tick labels                       |
  | `subtitle` | 7.5       | Grey metadata line below title    |
  | `legend`   | 6.5       | Legend entries and title          |
  | `annot`    | 5.5       | In-cell heatmap annotations       |

- Never hardcode font sizes directly. Always reference `FS["key"]`.

## Color Palette

A custom warm colormap called `SUNSET`, defined as:

```python
LinearSegmentedColormap.from_list("sunset", [
    "#1B0504", "#5C1008", "#9C2810", "#CC5A20",
    "#E09040", "#EDBC70", "#F5DCA8",
])
```

- **Heatmaps:** Use `SUNSET` directly with `imshow(cmap=SUNSET)`.
- **Line plots / discrete series:** Sample from `SUNSET` avoiding near-black and near-white extremes:
  ```python
  colors = [SUNSET(0.82 - 0.65 * i / (n - 1)) for i in range(n)]   # plotstyle.sunset_colors(n)
  ```
- **Single accent color:** `#CC5A20` (the mid-orange from the palette; `plotstyle.ACCENT`).
- **Do not** use viridis, plasma, or other default colormaps. Keep the whole paper in one warm palette.

## Axes and Spines

- **Top and right spines:** off.
- **Left and bottom spines:** visible, light grey (`#cccccc`), 0.6pt weight.
- **Grid:** off by default. When needed (e.g., log-scale reference lines), use `color="#e8e8e8", lw=0.4` — barely visible.
- **Ticks:** inward direction, dark grey (`#333333`), major size 3pt, minor size 1.5pt.
- **Line width:** default 1.2pt for data lines.

## Titles and Subtitles

Every figure has a two-part title block (`plotstyle.titles(fig, claim, metadata)`):

1. **`fig.suptitle(...)`** — bold, `FS["title"]` pt, dark (`#111111`). This is the figure's main claim.
2. **`fig.text(...)`** — regular weight, `FS["subtitle"]` pt, medium grey (`#555555`), centered
   below the title. This carries metadata: sample sizes, minutes range, seed. Uses `|` as separator.

Do **not** put figure titles in `ax.set_title()` — that's reserved for panel labels.

## Panel Labels

For multi-panel figures, panel labels are **merged into the axes title** (`plotstyle.panel(ax, i, name)`):

```python
ax.set_title(f"({chr(97 + i)})  {name}", fontsize=FS["title"] - 1,
             fontweight="semibold", color="#222222", pad=5)
```

This produces "(a)  ...", "(b)  ...". Do not use a separate floating text element for panel labels.

## Legends

- `frameon=False` — no box around the legend (`plotstyle.legend(ax)`).
- `labelcolor="#333333"` — slightly softer than black.
- Use `FS["legend"]` for font size.
- Many categories: place outside the axes and reserve room. Few categories: in-axes is fine.

## Colorbars

- `fig.colorbar(im, ax=..., shrink=0.75, pad=0.03, aspect=25)`; never `fig.add_axes([...])`.
- `cbar.outline.set_visible(False)`, tick length 0.

## Layout

- Reserve title headroom with `fig.tight_layout(rect=[0, 0, 1, top])`; do not use `constrained_layout`
  (it conflicts with manual `fig.text()` positioning).
- Do **not** set `left`/`bottom` — let `bbox_inches="tight"` handle outer margins.

## Math and Special Characters

- Greek/math: always matplotlib math mode (`r"$\beta$"`, `r"$N$"`). Renders under both `USE_TEX` settings.
- En-dashes: Unicode `–` (U+2013), not `--`. Separators in subtitles: `|` with surrounding spaces.

## Checklist for New Plots

1. Width from `TEXTWIDTH` (full) or `TEXTWIDTH * 0.55` (half).
2. Colors from `SUNSET`, not default colormaps.
3. All font sizes from `FS`.
4. `plotstyle.titles(...)`, not `ax.set_title`, for figure-level titles.
5. Panel labels merged into axes titles if multi-panel.
6. Save via `plotstyle.save(fig, stem)` — writes PDF and PNG with `**SAVE_KW`.
7. No `constrained_layout`. Use `tight_layout(rect=...)` + `bbox_inches="tight"`.
