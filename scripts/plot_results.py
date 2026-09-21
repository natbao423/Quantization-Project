"""Render the headline results as images for Docs, Sheets and slides.

    python scripts/plot_results.py                  # every figure
    python scripts/plot_results.py --only fig1 fig3
    python scripts/plot_results.py --format svg     # vector, for a paper

Writes to results/figures/. Every figure gets a CSV twin holding exactly the
numbers it plots. That is the table view: a static image has no hover, so any
value a reader cannot read off the chart must still be reachable, and the CSV
also pastes straight into Google Sheets if you would rather build a native
chart there. Images carry no title or description by default, so they sit
under a caption of your own; the text for each - including caveats the image
cannot show - is written to captions.txt, and --titles draws it on the image.
A figures.meta.json records the commit and the source files, so a pasted image
can be traced back to the data that drew it.

Reads only committed results files. A figure whose data is missing is skipped
with a message rather than drawn from nothing.

Design rules, briefly, because they are why the charts look the way they do:
  - At most three colored series per panel. The first three palette slots are
    the largest set that stays distinguishable under colour-vision deficiency
    when any two can sit side by side, which small multiples allow.
  - One y-axis per panel, never two. Different measures get different panels.
  - When the point is "these several things behave alike and this one does
    not", the alike ones are drawn in one gray and only the exception gets a
    colour. That is the fig 1 story, and giving each gray line its own hue
    would bury it.
  - Text is always in ink, never in a series colour.
  - The 23-bit row is plotted as "FP32" and the noise floor it measures is
    shaded, so a difference smaller than the floor cannot be read as a result.
"""

import argparse
import csv
import pathlib
import statistics as st
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")                      # render to files; no window needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from fpbench.run_metadata import describe_run, git_state, save_metadata

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"

# Palette: the data-viz skill's documented reference instance, light mode.
# Validated with its validate_palette.py: slots 1-3 pass every hard gate on the
# all-pairs test (worst CVD dE 9.2, normal-vision dE 24.0). Slot 3 sits at
# 2.74:1 against the surface, below 3:1, so wherever it appears it carries a
# distinct marker shape and direct labels, and its values are in the CSV twin.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # titles
INK2 = "#52514e"         # every other piece of text, 7.7:1 on the surface
GRAY = "#898781"         # de-emphasised marks, 3.5:1
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"         # noise-floor shading
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

BITS = [23, 10, 7, 5, 4, 3, 2, 1]
XTICKS = ["FP32", "10", "7", "5", "4", "3", "2", "1"]
PANELS = [("elementwise", "Per-element exponents"), ("bfp16", "Block floating point, block 16")]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def num(v):
    """Float, or None for a blank / missing cell."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def summary(*names):
    """(format, target, bits) -> row, merged across summarize_curves outputs."""
    out = {}
    for name in names:
        for r in read(name):
            out[(r["format"], r["target"], int(r["bits"]))] = r
    return out


def by_cell(rows, *keys):
    """Group per-seed rows by the given columns; `bits` is parsed as int."""
    g = defaultdict(list)
    for r in rows:
        g[tuple(int(r[k]) if k == "bits" else r[k] for k in keys)].append(r)
    return g


def col(cells, key, column, how=st.median):
    """Aggregate one column over the seeds in a cell, or None if absent."""
    vals = [num(r[column]) for r in cells.get(key, []) if num(r[column]) is not None]
    return how(vals) if vals else None


def need(*names):
    missing = [n for n in names if not (DATA / n).exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "axes.titlecolor": INK2, "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False,
    })


def key(color, label, marker="o", lw=2):
    """A legend entry drawn like the series it names."""
    return Line2D([0], [0], color=color, lw=lw, marker=marker, markersize=6,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, label=label,
                  linestyle="-" if lw else "none")


def figure(ncols, title, subtitle, keys, footer):
    """The plots and a legend row above them; the words go to captions.txt.

    By default the image carries no title, description or source line, so it
    can sit under a caption of your own in a document. That text still has to
    exist somewhere - several captions carry a caveat the image cannot show,
    such as fig 6 not being a like-for-like comparison - so it is attached to
    the figure and written to captions.txt by main(). --titles draws it on the
    image instead.
    """
    width = 8.2 if ncols == 2 else 6.6
    if not ARGS.titles:
        fig, axes = plt.subplots(1, ncols, figsize=(width, 3.35), squeeze=False)
        fig.subplots_adjust(left=0.085 if ncols == 2 else 0.1, right=0.975,
                            top=0.8 if ncols == 2 else 0.86, bottom=0.155,
                            wspace=0.26)
        fig.legend(handles=keys, loc="upper left", bbox_to_anchor=(0.008, 0.985),
                   ncol=len(keys), fontsize=8.5, labelcolor=INK2,
                   handlelength=2.2, columnspacing=1.8, borderaxespad=0)
        fig.caption = (title, subtitle, footer)
        return fig, list(axes[0])

    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.5), squeeze=False)
    fig.subplots_adjust(left=0.085 if ncols == 2 else 0.1, right=0.975,
                        top=0.66 if ncols == 2 else 0.715, bottom=0.2, wspace=0.26)
    fig.caption = (title, subtitle, footer)
    wrap = int(width * 15.5)
    fig.text(0.015, 0.965, title, fontsize=12.5, fontweight="semibold",
             color=INK, va="top")
    fig.text(0.015, 0.905, textwrap.fill(subtitle, wrap), fontsize=9,
             color=INK2, va="top", linespacing=1.35)
    fig.legend(handles=keys, loc="upper left", bbox_to_anchor=(0.008, 0.795),
               ncol=len(keys), fontsize=8.5, labelcolor=INK2, handlelength=2.2,
               columnspacing=1.8, borderaxespad=0)
    fig.text(0.015, 0.025, textwrap.fill(footer, int(width * 19)), fontsize=7,
             color=INK2, va="bottom", linespacing=1.3)
    return fig, list(axes[0])


def bits_axis(ax):
    ax.set_xticks(range(len(BITS)), XTICKS)
    ax.set_xlim(-0.3, len(BITS) - 0.7)
    ax.set_xlabel("mantissa bits  (fewer to the right)")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=4)


def line(ax, ys, color, marker="o", ms=6.5, z=3, lw=2):
    """2px line, markers with a 2px surface ring so crossings stay legible.

    lw=0 draws markers only - for a series that lands exactly on another, so
    the one underneath stays visible and the coincidence reads as a result
    rather than as a missing line.
    """
    pts = [(i, y) for i, y in enumerate(ys) if y is not None]
    if not pts:
        return
    xs, yv = zip(*pts)
    ax.plot(xs, yv, color=color, lw=lw, marker=marker, ms=ms,
            mec=SURFACE, mew=1.6, zorder=z, linestyle="-" if lw else "none")


def ribbon(ax, lo, hi, color):
    """Seed spread as a ~12% wash of the series hue: a range, not a block."""
    pts = [(i, a, b) for i, (a, b) in enumerate(zip(lo, hi)) if a is not None]
    xs, a, b = zip(*pts)
    ax.fill_between(xs, a, b, color=color, alpha=0.12, lw=0, zorder=1)


def label(ax, x, y, text, dx=-8, dy=8, ha="right", va="bottom"):
    """A direct label in ink beside a mark - never in the mark's colour."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=8, color=INK2, ha=ha, va=va, zorder=6)


def floor_band(ax, lo, hi, text="noise floor", x=None, above=False):
    """Shade the noise floor and name it where no data runs through the text."""
    ax.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    x = len(BITS) - 1 if x is None else x
    y, dy, va = (hi, 3, "bottom") if above else (lo, 3, "bottom")
    ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points",
                fontsize=7.5, color=INK2, ha="right", va=va, zorder=6)


def shown(path):
    """Repo-relative for display, or absolute when --out-dir is elsewhere."""
    path = pathlib.Path(path).resolve()
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def save(fig, path):
    fig.savefig(path, dpi=ARGS.dpi)
    plt.close(fig)


def tidy(fig_id, panel, series, bits, value, lo=None, hi=None):
    """One row of a figure's CSV twin."""
    return {"figure": fig_id, "panel": panel, "series": series, "bits": bits,
            "value": value, "min": lo, "max": hi}


# --------------------------------------------------------------------------
# figures - each returns the tidy rows it plotted
# --------------------------------------------------------------------------

CNN_FILES = ("mnist_cnn_summary.csv", "mnist_cnn_summary_grad_grad_sr.csv")


def fig1():
    """The headline: only the stored parameter is expensive."""
    need(*CNN_FILES)
    s = summary(*CNN_FILES)
    kl = lambda f, t, b: num(s.get((f, t, b), {}).get("kl"))
    computation = ["input", "activation", "weight_master", "grad"]

    gaps = [kl(f, "weight", b) / kl(f, "weight_master", b)
            for f, _ in PANELS for b in (5, 4, 3, 2, 1)]
    fig, axes = figure(
        2, "Only quantizing the stored weight is expensive",
        "Small CNN on MNIST: KL divergence from the FP32 model trained at the same seed "
        "(log scale, median of 3 seeds). Four ways of quantizing a computation land "
        f"together; re-rounding the weights themselves sits {min(gaps):.0f}–"
        f"{max(gaps):.0f}× higher at 5 bits and below.",
        [key(S1, "weight  (re-rounded every step, no FP32 master)"),
         key(GRAY, "input,  activation,  weight_master,  grad")],
        "Source: results/data/mnist_cnn_summary.csv; the grad line is from the separate "
        "sweep in mnist_cnn_summary_grad_grad_sr.csv, with its own FP32 references. "
        f"Shaded band: the 23-bit (FP32) spread, i.e. cuDNN run-to-run noise. Commit {SHA}.")

    rows = []
    for ax, (fmt, title) in zip(axes, PANELS):
        for t in computation:
            ys = [kl(fmt, t, b) for b in BITS]
            line(ax, ys, GRAY, ms=5.5, z=2)
            rows += [tidy("fig1", fmt, t, b, y) for b, y in zip(BITS, ys)]
        ys = [kl(fmt, "weight", b) for b in BITS]
        line(ax, ys, S1, z=4)
        rows += [tidy("fig1", fmt, "weight", b, y) for b, y in zip(BITS, ys)]

        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(3e-5, 2)
        floor = max(kl(fmt, t, 23) for t in computation + ["weight"])
        floor_band(ax, 3e-5, floor)

        w, m = kl(fmt, "weight", 1), kl(fmt, "weight_master", 1)
        ax.annotate("", xy=(7.22, w), xytext=(7.22, m),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9,
                                    shrinkA=2, shrinkB=2))
        label(ax, 7.22, (w * m) ** 0.5, f"{w / m:.0f}×", dx=-5, dy=0, va="center")
        ax.set_title(title)
        ax.set_ylabel("KL from FP32  (log)")
        bits_axis(ax)
        ax.set_xlim(-0.3, 7.45)
    return fig, rows


def fig2():
    """Deferred versus discarded: same weight motion, opposite outcome."""
    need(CNN_FILES[0])
    s = summary(CNN_FILES[0])
    get = lambda t, b, c: num(s.get(("elementwise", t, b), {}).get(c))

    fig, axes = figure(
        2, "A master copy moves fewer weights, yet trains far better",
        "Small CNN on MNIST, per-element exponents, median of 3 seeds. Without a master, a "
        "weight that does not move has had its update discarded for good; with one, the "
        "update was only deferred into the FP32 copy and lands later.",
        [key(S1, "weight  (no master)"), key(S2, "weight_master  (FP32 master kept)")],
        "Source: results/data/mnist_cnn_summary.csv. Left: fraction of quantized weight "
        "elements whose value changed per optimizer step, whole-run average (FP32 omitted: "
        "every weight moves). Right: shaded band is FP32 accuracy ±0.004, the noise band. "
        f"Commit {SHA}.")

    rows = []
    left, right = axes
    for t, c in (("weight", S1), ("weight_master", S2)):
        ys = [None] + [get(t, b, "upd_survive") for b in BITS[1:]]
        line(left, ys, c, z=4 if t == "weight" else 3)
        rows += [tidy("fig2", "update_survival", t, b, y) for b, y in zip(BITS, ys)]
        ys = [get(t, b, "final_acc") for b in BITS]
        line(right, ys, c, z=4 if t == "weight" else 3)
        rows += [tidy("fig2", "accuracy", t, b, y) for b, y in zip(BITS, ys)]

    left.set_yscale("log")
    left.yaxis.set_minor_locator(NullLocator())
    left.set_ylim(0.02, 1)
    left.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1],
                    ["2%", "5%", "10%", "20%", "50%", "100%"])
    label(left, 7, get("weight", 1, "upd_survive"),
          f"{get('weight', 1, 'upd_survive'):.1%}", dy=7)
    label(left, 7, get("weight_master", 1, "upd_survive"),
          f"{get('weight_master', 1, 'upd_survive'):.1%}", dy=-7, va="top")
    left.set_title("Weights that moved per step  (log)")

    base = st.mean(get(t, 23, "final_acc") for t in ("weight", "weight_master"))
    right.axhspan(base - 0.004, base + 0.004, color=BAND, lw=0, zorder=0)
    right.annotate("FP32 ± noise", (0, base - 0.004), xytext=(2, -3),
                   textcoords="offset points", fontsize=7.5, color=INK2,
                   ha="left", va="top", zorder=6)
    right.set_ylim(0.9, 0.995)
    right.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    label(right, 7, get("weight", 1, "final_acc"),
          f"{get('weight', 1, 'final_acc'):.1%}", dy=-7, va="top")
    label(right, 7, get("weight_master", 1, "final_acc"),
          f"{get('weight_master', 1, 'final_acc'):.1%}", dy=-9, va="top")
    right.set_title("Validation accuracy")
    for ax in axes:
        bits_axis(ax)
    return fig, rows


def fig3():
    """Gradient quantization is free, in both batching regimes."""
    mlp_file = "train_at_vary_precision_weight_grad_grad_sr.csv"
    need(*CNN_FILES, mlp_file)
    s = summary(*CNN_FILES)
    cnn = lambda t, b: num(s.get(("bfp16", t, b), {}).get("final_acc"))
    cells = by_cell(read(mlp_file), "format", "target", "bits")
    mlp = lambda t, b: col(cells, ("bfp16", t, b), "r2")

    fig, axes = figure(
        2, "Quantizing gradients costs nothing, with or without minibatch noise",
        "Block floating point, block 16. Gradients are rounded between backward() and the "
        "optimizer step; weights stay FP32. Stochastic rounding (grad_sr) lands on the same "
        "line as round-to-nearest (grad): there was no gap for it to close.",
        [key(S1, "weight  (no master)"), key(S2, "grad  (round to nearest)"),
         key(S3, "grad_sr  (stochastic, markers on the grad line)", marker="D", lw=0)],
        "Left: results/data/mnist_cnn_summary*.csv, minibatch SGD, median of 3 seeds. "
        "Right: results/data/train_at_vary_precision_weight_grad_grad_sr.csv, full-batch SGD, "
        f"median of 10 seeds. Commit {SHA}.")

    rows = []
    for ax, name, get, ylab in ((axes[0], "CNN on MNIST  (minibatch)", cnn, "validation accuracy"),
                                (axes[1], "MLP regression  (full-batch)", mlp, "R²")):
        for t, c, m, ms, z, lw in (("weight", S1, "o", 6.5, 3, 2),
                                   ("grad", S2, "o", 8, 4, 2),
                                   ("grad_sr", S3, "D", 4.5, 5, 0)):
            ys = [get(t, b) for b in BITS]
            line(ax, ys, c, marker=m, ms=ms, z=z, lw=lw)
            rows += [tidy("fig3", name, t, b, y) for b, y in zip(BITS, ys)]
        label(ax, 7, get("grad", 1), "grad, grad_sr", dy=-8, va="top")
        w1 = get("weight", 1)
        label(ax, 7, w1, f"weight  {w1:.1%}" if get is cnn else f"weight  R² {w1:.2f}",
              dy=-8, va="top")
        ax.set_title(name)
        ax.set_ylabel(ylab)
        bits_axis(ax)
    axes[0].set_ylim(0.78, 1.0)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_ylim(0, 1.05)
    return fig, rows


def fig4():
    """How much of the gradient block floating point actually destroys."""
    mlp_file = "train_at_vary_precision_weight_grad_grad_sr.csv"
    need(CNN_FILES[1], mlp_file)
    s = summary(CNN_FILES[1])
    cells = by_cell(read(mlp_file), "format", "target", "bits")
    killed_cnn = lambda f, b: (None if num(s.get((f, "grad", b), {}).get("grad_survive")) is None
                               else 1 - num(s[(f, "grad", b)]["grad_survive"]))
    killed_mlp = lambda f, b: (None if col(cells, (f, "grad", b), "grad_survive", st.mean) is None
                               else 1 - col(cells, (f, "grad", b), "grad_survive", st.mean))

    fig, axes = figure(
        1, "Block floating point destroys up to 60% of the gradient",
        "Fraction of nonzero gradient elements rounded to exactly zero, per step, round "
        "to nearest. Figure 3 shows accuracy and R² did not move at any of these points. "
        "Per-element exponents destroy nothing: the grid follows each value down.",
        [key(S1, "CNN, BFP-16"), key(S2, "MLP, BFP-16"),
         key(GRAY, "both models, per-element")],
        "Sources: results/data/mnist_cnn_summary_grad_grad_sr.csv (3 seeds) and "
        f"train_at_vary_precision_weight_grad_grad_sr.csv (10 seeds). Commit {SHA}.")
    ax = axes[0]
    rows = []
    elem = [killed_cnn("elementwise", b) for b in BITS]
    line(ax, elem, GRAY, ms=5.5, z=2)
    rows += [tidy("fig4", "cnn", "elementwise", b, y) for b, y in zip(BITS, elem)]
    for name, get, c in (("cnn", killed_cnn, S1), ("mlp", killed_mlp, S2)):
        ys = [get("bfp16", b) for b in BITS]
        line(ax, ys, c, z=4)
        rows += [tidy("fig4", name, "bfp16", b, y) for b, y in zip(BITS, ys)]
        label(ax, 7, ys[-1], f"{ys[-1]:.0%}", dy=8)
    ax.set_ylim(-0.02, 0.7)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("gradient elements destroyed")
    bits_axis(ax)
    return fig, rows


def fig5():
    """The MLP's cliff: initialization decides which side of it a run lands."""
    need("train_at_vary_precision.csv")
    cells = by_cell(read("train_at_vary_precision.csv"), "target", "bits")

    fig, axes = figure(
        1, "The MLP's weight cliff depends on initialization",
        "2-layer MLP on synthetic regression, per-element exponents, weights re-rounded "
        "every step with no master. Line is the median of 10 seeds; the shaded range is "
        "the lowest to highest seed. At 4 bits the same configuration spans nearly the "
        "whole scale.",
        [key(S1, "weight"), key(S2, "input")],
        f"Source: results/data/train_at_vary_precision.csv, 2000 full-batch epochs. Commit {SHA}.")
    ax = axes[0]
    rows = []
    for t, c in (("weight", S1), ("input", S2)):
        med = [col(cells, (t, b), "r2") for b in BITS]
        lo = [col(cells, (t, b), "r2", min) for b in BITS]
        hi = [col(cells, (t, b), "r2", max) for b in BITS]
        ribbon(ax, lo, hi, c)
        line(ax, med, c, z=4 if t == "weight" else 3)
        rows += [tidy("fig5", "mlp", t, b, m, a, z) for b, m, a, z in zip(BITS, med, lo, hi)]
    i4 = BITS.index(4)
    lo4, hi4 = col(cells, ("weight", 4), "r2", min), col(cells, ("weight", 4), "r2", max)
    ax.annotate("", xy=(i4 - 0.12, hi4), xytext=(i4 - 0.12, lo4),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    label(ax, i4 - 0.12, (lo4 + hi4) / 2, f"seeds span\n{lo4:.2f} to {hi4:.2f}",
          dx=-6, dy=0, va="center")
    label(ax, 7, col(cells, ("input", 1), "r2"), "input", dy=-8, va="top")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²  (fraction of target variance explained)")
    bits_axis(ax)
    return fig, rows


def fig6():
    """The transformer, with its confound stated on the figure."""
    need("char_transformer_curves.csv")
    last = {}
    for r in read("char_transformer_curves.csv"):
        k = (r["format"], r["target"], int(r["bits"]), r["seed"])
        if k not in last or int(r["step"]) > int(last[k]["step"]):
            last[k] = r
    cells = by_cell(last.values(), "format", "target", "bits")

    fig, axes = figure(
        2, "Transformer: activations hold to about 4 bits; weights degrade from 7",
        "Character-level transformer on Tiny Shakespeare, median validation perplexity of "
        "3 seeds after 4,000 steps (lower is better). This sweep has no weight_master "
        "condition, so the weight curve mixes representation error with update-vanishing "
        "and is not a like-for-like comparison with activations.",
        [key(S1, "activation"), key(S2, "weight  (no master)")],
        "Source: results/data/char_transformer_curves.csv. The 'both' condition overlaps "
        f"'weight' and is omitted; it is in the CSV twin. Commit {SHA}.")
    rows = []
    for ax, (fmt, title) in zip(axes, PANELS):
        base = col(cells, (fmt, "activation", 23), "val_ppl")
        ax.axhline(base, color=AXIS, lw=0.9, zorder=1)
        ax.annotate("FP32", (len(BITS) - 1, base), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.5, color=INK2,
                    ha="right", va="bottom")
        for t, c in (("activation", S1), ("weight", S2), ("both", None)):
            ys = [col(cells, (fmt, t, b), "val_ppl") for b in BITS]
            if c:
                line(ax, ys, c, z=4 if t == "activation" else 3)
            rows += [tidy("fig6", fmt, t, b, y) for b, y in zip(BITS, ys)]
        ax.set_ylim(4, 29)
        # Direct labels only where the two series separate at the right edge.
        # Under BFP they converge at 1 bit, and stacking labels there would
        # detach them from their lines; the legend carries identity instead.
        if fmt == "elementwise":
            for t in ("activation", "weight"):
                y = col(cells, (fmt, t, 1), "val_ppl")
                # activation sits between the two lines at its own height
                label(ax, 7, y, t, dx=-10, dy=0 if t == "activation" else 7,
                      va="center" if t == "activation" else "bottom")
        ax.set_title(title)
        ax.set_ylabel("validation perplexity")
        bits_axis(ax)
    return fig, rows


# --------------------------------------------------------------------------
# the gradient sweep in detail: the last run, with --evals-per-epoch 10
# --------------------------------------------------------------------------

T98 = "mnist_cnn_summary_grad_t98.csv"
T98_CURVES = "mnist_cnn_curves_grad_t98.csv"
VAL_IMAGES = 5000          # train_mnist_cnn.VAL_SIZE

# Same colours and markers as fig 3, so grad and grad_sr read as the same
# entities in every figure. grad_sr's aqua is below 3:1 on the surface, so it
# always carries the diamond marker and a direct label, and sits in the CSV.
GRAD = (("grad", S2, "o", 6.5, 4), ("grad_sr", S3, "D", 5.5, 5))


def grad_keys():
    return [key(S2, "grad  (round to nearest)"),
            key(S3, "grad_sr  (stochastic)", marker="D")]


def t98_footer(extra=""):
    return ("Source: results/data/mnist_cnn_summary_grad_t98.csv, the gradient sweep "
            "run with --evals-per-epoch 10. Small CNN on MNIST, median of 3 seeds. "
            f"{extra}Commit {SHA}.")


def end_labels(ax, items, gap_px=24):
    """Label each series just right of its 1-bit point.

    Nothing is plotted to the right of the last point, so a label placed there
    can never sit on a line - labels placed beside the point collided whenever
    the two series crossed in the last segment. The x-axis is widened to make
    room. Two endpoints closer than a label is tall share one two-line label,
    higher value on top, rather than being pushed apart from their points.
    """
    x = len(BITS) - 1
    ax.set_xlim(-0.3, x + 2.5)
    items = sorted(((y, t) for y, t in items if y is not None), reverse=True)
    px = lambda y: ax.transData.transform((x, y))[1]
    style = dict(xytext=(9, 0), textcoords="offset points", fontsize=8,
                 color=INK2, ha="left", va="center", zorder=6)
    if len(items) == 2 and abs(px(items[0][0]) - px(items[1][0])) < gap_px:
        mid = (px(items[0][0]) + px(items[1][0])) / 2
        xpix = ax.transData.transform((x, items[0][0]))[0]
        ymid = ax.transData.inverted().transform((xpix, mid))[1]
        ax.annotate(chr(10).join(t for _, t in items), (x, ymid),
                    linespacing=1.25, **style)
    else:
        for y, t in items:
            ax.annotate(t, (x, y), **style)


def grad_panels(fid, metric, title, subtitle, ylabel, show, *, log=False,
                ylim=None, band=None, pct=None, yticks=None, footer_extra=""):
    """Two panels (per-element | BFP-16), grad against grad_sr, for one metric.

    band="floor" shades from the bottom of the axis up to the 23-bit spread,
    for metrics that measure distance from FP32 and so should be zero there.
    band="spread" shades the range the two 23-bit runs landed in, for metrics
    that do not have a zero.
    """
    need(T98)
    s = summary(T98)
    get = lambda f, t, b: num(s.get((f, t, b), {}).get(metric))
    fig, axes = figure(2, title, subtitle, grad_keys(), t98_footer(footer_extra))
    rows = []
    for ax, (fmt, ptitle) in zip(axes, PANELS):
        for t, c, m, ms, z in GRAD:
            ys = [get(fmt, t, b) for b in BITS]
            line(ax, ys, c, marker=m, ms=ms, z=z)
            rows += [tidy(fid, fmt, t, b, y) for b, y in zip(BITS, ys)]
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(NullLocator())
        if ylim:
            ax.set_ylim(*ylim)
        fp32 = [get(fmt, t, 23) for t, *_ in GRAD]
        if band == "floor":
            floor_band(ax, ax.get_ylim()[0], max(fp32))
        elif band == "spread":
            ax.axhspan(min(fp32), max(fp32), color=BAND, lw=0, zorder=0)
            ax.annotate("FP32 run-to-run", (0, min(fp32)), xytext=(0, -4),
                        textcoords="offset points", fontsize=7.5, color=INK2,
                        ha="left", va="top", zorder=6)
        if yticks is not None:
            ax.set_yticks(yticks)
        if pct:
            ax.yaxis.set_major_formatter(lambda v, _: format(v, pct))
        ax.set_title(ptitle)
        ax.set_ylabel(ylabel)
        bits_axis(ax)
        end_labels(ax, [(get(fmt, t, 1), f"{t}  {show(get(fmt, t, 1))}") for t, *_ in GRAD])
    return fig, rows


def fig7():
    """KL divergence from FP32 under gradient quantization."""
    weight_kl = None
    if (DATA / CNN_FILES[0]).exists():
        weight_kl = num(summary(CNN_FILES[0]).get(("elementwise", "weight", 1), {}).get("kl"))
    scale = (f" For scale, quantizing the weights themselves reaches {weight_kl:.2f} at 1 bit "
             "(figure 1)." if weight_kl else "")
    return grad_panels(
        "fig7", "kl", "Quantizing gradients barely moves the trained model",
        "KL divergence of each model's predictions from the FP32 model trained at the same "
        "seed, log scale. It rises as bits drop, most under block floating point, but stays "
        "within about 15 times FP32's own run-to-run noise." + scale,
        "KL from FP32  (log)", lambda v: f"{v:.4f}", log=True, ylim=(3e-5, 5e-3),
        band="floor",
        footer_extra="Shaded band: the 23-bit (FP32) spread, i.e. run-to-run noise. ")


def fig8():
    """Centered relative logit error: the part of the error softmax can see."""
    return grad_panels(
        "fig8", "logit_rel_err_c",
        "The logit error that can change a prediction stays within a few percent",
        "Relative error of each model's logits against the FP32 model trained at the same "
        "seed, after removing each image's mean logit. Softmax ignores a shift applied to "
        "all ten logits, so this centered figure is the part that can change a prediction.",
        "relative logit error  (centered)", lambda v: f"{v:.1%}", ylim=(0, 0.085),
        band="floor", pct=".0%",
        footer_extra="Shaded band: the 23-bit (FP32) spread, i.e. run-to-run noise. ")


def fig9():
    """How much of the raw logit error is a harmless per-image shift."""
    need(T98)
    s = summary(T98)
    get = lambda f, b, c: num(s.get((f, "grad", b), {}).get(c))
    fig, axes = figure(
        2, "Most of block floating point's logit error is a shift softmax ignores",
        "Round-to-nearest gradients. The raw relative logit error counts a shift applied "
        "equally to all ten logits of an image, which cannot change any prediction; the "
        "centered error removes it. Under BFP at 1 bit the raw figure is about 11 times the "
        "centered one, so nearly all of it is that harmless shift.",
        [key(S2, "centered  (can change predictions)"),
         key(GRAY, "raw  (includes the shift)")],
        t98_footer("Metric trap also described for weights in the README: always read the "
                   "centered figure. "))
    rows = []
    for ax, (fmt, ptitle) in zip(axes, PANELS):
        for c, color, z, name in (("logit_rel_err", GRAY, 3, "raw"),
                                  ("logit_rel_err_c", S2, 4, "centered")):
            ys = [get(fmt, b, c) for b in BITS]
            line(ax, ys, color, z=z)
            rows += [tidy("fig9", fmt, name, b, y) for b, y in zip(BITS, ys)]
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0.008, 1.2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}" if v >= 0.01 else f"{v:.1%}")
        raw, cen = get(fmt, 1, "logit_rel_err"), get(fmt, 1, "logit_rel_err_c")
        # Values only: the names are long enough to run into the next panel, and
        # the gray and orange markers beside them, plus the legend, carry identity.
        ax.set_title(ptitle)
        ax.set_ylabel("relative logit error  (log)")
        bits_axis(ax)                      # before end_labels: it resets the x-range
        end_labels(ax, [(raw, f"{raw:.1%}"), (cen, f"{cen:.1%}")])
    return fig, rows


def fig10():
    """Predictions that differ from the FP32 model's."""
    return grad_panels(
        "fig10", "disagree", "Rounding gradients changes almost none of the predictions",
        f"Share of the {VAL_IMAGES:,} validation images whose predicted digit differs from "
        "the FP32 model trained at the same seed. At 1 bit that is 10 to 15 images; FP32's "
        "own run-to-run disagreement is 3 to 6.",
        "predictions that differ from FP32", lambda v: f"{v:.2%}", ylim=(0, 0.0045),
        band="floor", pct=".1%", yticks=[0, 0.001, 0.002, 0.003, 0.004],
        footer_extra="Shaded band: the 23-bit (FP32) spread, i.e. run-to-run noise. ")


def fig11():
    """Cosine similarity between the rounded and unrounded gradient."""
    return grad_panels(
        "fig11", "grad_cos", "Rounded gradients still point almost the same way",
        "Cosine similarity between each step's rounded gradient and the unrounded gradient "
        "at the same step, averaged over training; 1 means an identical direction. "
        "Stochastic rounding scores lower by design, since it trades bias for step-to-step "
        "noise: do not use this to rank the two rounding rules.",
        "cosine similarity to unrounded gradient", lambda v: f"{v:.4f}",
        ylim=(0.95, 1.002))


def fig12():
    """Epochs to reach 98% validation accuracy, with the seed spread."""
    need(T98, T98_CURVES)
    per_seed = {}
    for r in read(T98_CURVES):
        per_seed[(r["format"], r["target"], int(r["bits"]), r["seed"])] = num(r["epochs_to_98"])
    cells = defaultdict(list)
    for (f, t, b, _), v in per_seed.items():
        if v is not None:
            cells[(f, t, b)].append(v)
    fig, axes = figure(
        2, "Rounding gradients does not slow training down",
        "Epochs until validation accuracy first touches 98%, measured to a tenth of an "
        "epoch, median of 3 seeds. The shaded band is the range the FP32 runs landed in. "
        "No configuration is slower than FP32. Every configuration, FP32 included, "
        "reached 90% at 0.1 epochs and 95% at 0.3.",
        grad_keys(),
        t98_footer("Stochastic BFP at 1-3 bits first touches 98% below FP32's range, but "
                   "these are brief touches, not faster training: at 1 bit, two seeds touched "
                   "98% at 1.4-1.5 epochs yet were below it again at the end of epochs 2 and 3, "
                   "and their epoch-end accuracy tracks FP32's almost exactly. Read this as no "
                   "slowdown, not as a speedup. Per-seed fastest and slowest are in the CSV "
                   "twin. "))
    rows = []
    for ax, (fmt, ptitle) in zip(axes, PANELS):
        fp32 = cells[(fmt, "grad", 23)] + cells[(fmt, "grad_sr", 23)]
        ax.axhspan(min(fp32), max(fp32), color=BAND, lw=0, zorder=0)
        ax.annotate("FP32 range", (0, min(fp32)), xytext=(0, -4),
                    textcoords="offset points", fontsize=7.5, color=INK2,
                    ha="left", va="top", zorder=6)
        for t, c, m, ms, z in GRAD:
            med = [st.median(cells[(fmt, t, b)]) if cells[(fmt, t, b)] else None for b in BITS]
            lo = [min(cells[(fmt, t, b)]) if cells[(fmt, t, b)] else None for b in BITS]
            hi = [max(cells[(fmt, t, b)]) if cells[(fmt, t, b)] else None for b in BITS]
            line(ax, med, c, marker=m, ms=ms, z=z)
            rows += [tidy("fig12", fmt, t, b, v, a, z_)
                     for b, v, a, z_ in zip(BITS, med, lo, hi)]
        ax.set_ylim(0, 3.2)
        ax.set_title(ptitle)
        ax.set_ylabel("epochs to 98% accuracy")
        bits_axis(ax)
        end_labels(ax, [(st.median(cells[(fmt, t, 1)]),
                         f"{t}  {st.median(cells[(fmt, t, 1)]):.1f}") for t, *_ in GRAD])
    return fig, rows


def fig13():
    """Final validation cross-entropy."""
    return grad_panels(
        "fig13", "final_loss", "Validation loss is unchanged at every width",
        "Validation cross-entropy after the final (12th) epoch. Every point sits within a "
        "few thousandths of FP32, the same size as the difference between two FP32 runs.",
        "validation cross-entropy", lambda v: f"{v:.4f}", ylim=(0.043, 0.0495),
        band="spread",
        footer_extra="Shaded band: the range the two 23-bit (FP32) runs landed in. ")


def fig14():
    """Gradient elements destroyed, by rounding rule."""
    need(T98)
    s = summary(T98)
    killed = lambda f, t, b: (None if num(s.get((f, t, b), {}).get("grad_survive")) is None
                              else 1 - num(s[(f, t, b)]["grad_survive"]))
    fig, axes = figure(
        1, "Stochastic rounding saves only a few gradient elements",
        "Fraction of nonzero gradient elements rounded to exactly zero per step, under "
        "block floating point. The dither lets a few tiny elements through that "
        "round-to-nearest would zero, but the difference is small and, per figures 7 "
        "to 13, changes nothing downstream. Per-element exponents destroy none.",
        [key(S2, "grad  (round to nearest)"), key(S3, "grad_sr  (stochastic)", marker="D"),
         key(GRAY, "per-element, both rules")],
        t98_footer())
    ax = axes[0]
    rows = []
    elem = [killed("elementwise", "grad", b) for b in BITS]
    line(ax, elem, GRAY, ms=5.5, z=2)
    rows += [tidy("fig14", "elementwise", "grad", b, y) for b, y in zip(BITS, elem)]
    for t, c, m, ms, z in GRAD:
        ys = [killed("bfp16", t, b) for b in BITS]
        line(ax, ys, c, marker=m, ms=ms, z=z)
        rows += [tidy("fig14", "bfp16", t, b, y) for b, y in zip(BITS, ys)]
    ax.set_ylim(-0.02, 0.7)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("gradient elements destroyed")
    bits_axis(ax)
    end_labels(ax, [(killed("bfp16", t, 1), f"{t}  {killed('bfp16', t, 1):.0%}")
                    for t, *_ in GRAD])
    return fig, rows


FIGURES = {
    "fig1": (fig1, "fig1_stored_parameter"),
    "fig2": (fig2, "fig2_deferred_vs_discarded"),
    "fig3": (fig3, "fig3_gradients_free"),
    "fig4": (fig4, "fig4_gradient_annihilation"),
    "fig5": (fig5, "fig5_mlp_cliff"),
    "fig6": (fig6, "fig6_transformer"),
    "fig7": (fig7, "fig7_grad_kl"),
    "fig8": (fig8, "fig8_grad_logit_error"),
    "fig9": (fig9, "fig9_grad_logit_shift"),
    "fig10": (fig10, "fig10_grad_disagreement"),
    "fig11": (fig11, "fig11_grad_cosine"),
    "fig12": (fig12, "fig12_grad_time_to_98"),
    "fig13": (fig13, "fig13_grad_loss"),
    "fig14": (fig14, "fig14_grad_survival_by_rounding"),
}


def main():
    ARGS.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    made, skipped, captions = [], [], []
    for fid in ARGS.only or FIGURES:
        fn, stem = FIGURES[fid]
        try:
            fig, rows = fn()
        except FileNotFoundError as e:
            skipped.append(fid)
            print(f"skip {fid}: missing {e}", flush=True)
            continue
        img = ARGS.out_dir / f"{stem}.{ARGS.format}"
        title, subtitle, footer = fig.caption
        captions.append(f"{img.name}\nFigure {fid[3:]}. {title}. {subtitle}\n{footer}\n")
        save(fig, img)
        with (ARGS.out_dir / f"{stem}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        made.append(img.name)
        print(f"wrote {shown(img)}  (+ {stem}.csv)", flush=True)

    if captions and not ARGS.only:
        # Plain text rather than markdown: it pastes into Docs as-is.
        (ARGS.out_dir / "captions.txt").write_text("\n".join(captions), encoding="utf-8")
        print(f"wrote {shown(ARGS.out_dir / 'captions.txt')}", flush=True)
    save_metadata(ARGS.out_dir / "figures", describe_run(
        args=ARGS, extra={"status": "complete", "figures": made, "skipped": skipped}))
    print(f"\n{len(made)} figure(s) in {shown(ARGS.out_dir)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Render result figures for Docs, Sheets and slides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--only", nargs="+", choices=sorted(FIGURES), metavar="FIG",
                   help="figures to draw: " + ", ".join(FIGURES))
    p.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "results" / "figures")
    p.add_argument("--format", choices=["png", "svg", "pdf"], default="png",
                   help="png pastes into Docs and Sheets; svg or pdf for a paper")
    p.add_argument("--titles", action="store_true",
                   help="draw the title, description and source line on the "
                        "image. Off by default; the same text always goes to "
                        "captions.txt")
    p.add_argument("--dpi", type=int, default=220,
                   help="png resolution; 220 gives ~1,800px across, sharp at full page width")
    ARGS = p.parse_args()
    SHA = (git_state()["commit"] or "unknown")[:7]
    main()
