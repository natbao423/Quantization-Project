"""Collapse per-epoch curves into per-configuration summaries.

    python scripts/summarize_curves.py
    python scripts/summarize_curves.py --input results/data/mnist_cnn_curves_weight.csv

Defaults to results/data/mnist_cnn_curves.csv and writes the matching
_summary.csv. A narrowed sweep (--only) writes its own _curves_<tags>.csv, and
summarizing that lands in _summary_<tags>.csv rather than overwriting the
canonical summary.

Reports final-epoch and best-epoch numbers side by side. Final is the honest
frozen-budget number and is what the headline should use. Best is a check: if
a configuration peaked well before the budget, the final number is measuring
decline rather than precision.

Accuracy on 5,000 validation images has a standard error near 0.0016 at the
98.7% baseline, so gaps under roughly 0.004 are not measurable no matter how
many seeds are run. Four unbounded columns are summarized alongside it when
present: kl, disagree, logit_rel_err_c (all measured against the FP32 model
trained at the same seed) and upd_survive. Files written before those columns
existed still summarize; the extra tables are simply skipped.

Perplexity is deliberately absent. On a ten-class problem it is a monotone
transform of val_loss and adds nothing.
"""

import argparse
import csv
import json
import pathlib
import statistics as st
from collections import defaultdict

from fpbench.run_metadata import describe_run, save_metadata

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"

# Print order. "activation" sits next to "weight_master" because they are the
# pair that is actually comparable: both compute the gradient at a quantized
# point and apply it somewhere unquantized, so both isolate representation
# error. "weight" additionally discards sub-grid updates.
TARGETS = ["input", "activation", "weight_master", "weight", "both",
           "act_weight", "grad", "grad_sr"]

# First-hit times at sub-epoch resolution, written by sweeps run with
# --evals-per-epoch. Must match train_mnist_cnn.TIME_TO.
FINE = [f"epochs_to_{int(t * 100)}" for t in (0.90, 0.95, 0.98)]

# Unbounded metrics, reported only if the CSV carries them.
EXTRA = ["kl", "disagree", "logit_rel_err", "logit_rel_err_c",
         "upd_survive", "grad_survive", "grad_cos"]

# Accuracy thresholds for the time-to-target tables. One threshold cannot work
# for the whole sweep: 0.98 is unreachable for anything at or below 3 bits, and
# 0.90 is reached in the first epoch by everything that works at all. Reading
# across the three tables is the point. A row of "never" is a ceiling, a rising
# epoch count with no "never" is a slowdown, and the two are different claims.
TARGET_ACCS = (0.90, 0.95, 0.98)


def _repo_relative(path):
    """Repo-relative path for the metadata, or absolute if it lives outside.

    Both branches are needed: --input accepts a relative path, which must be
    resolved before it can be compared against ROOT, and it also accepts a file
    from anywhere on disk, which has no repo-relative form at all.
    """
    resolved = pathlib.Path(path).resolve()
    return (str(resolved.relative_to(ROOT))
            if resolved.is_relative_to(ROOT) else str(resolved))


def fnum(s):
    """Float, or None for a blank cell. A blank means the metric was not
    recorded for that condition, which is not the same as a recorded zero."""
    return float(s) if s not in (None, "") else None


def load(IN):
    """(format, target, bits, seed) -> list of per-epoch dicts, sorted."""
    runs = defaultdict(list)
    with IN.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for r in reader:
            key = (r["format"], r["target"], int(r["bits"]), int(r["seed"]))
            row = {"epoch": int(r["epoch"]),
                   "val_acc": float(r["val_acc"]),
                   "val_loss": float(r["val_loss"])}
            for k in EXTRA + FINE:
                if k in r:
                    row[k] = fnum(r[k])
            runs[key].append(row)
    for v in runs.values():
        v.sort(key=lambda d: d["epoch"])
    return runs, [k for k in EXTRA if k in cols], all(k in cols for k in FINE)


def epochs_to(curve, target):
    """First epoch reaching `target` accuracy, or None if it never does.

    Turns a saturated endpoint metric into an unsaturated one: low precision
    slows convergence rather than capping it, and a frozen 12-epoch budget
    reports only where each run happened to land.
    """
    for r in curve:
        if r["val_acc"] >= target:
            return r["epoch"]
    return None


def per_run(curve, extra, fine=False):
    final = curve[-1]
    best = max(curve, key=lambda r: r["val_acc"])
    out = {
        "final_acc": final["val_acc"],
        "final_loss": final["val_loss"],
        "best_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "min_loss": min(r["val_loss"] for r in curve),
        "epochs": final["epoch"],
    }
    for t in TARGET_ACCS:
        # Sweeps run with --evals-per-epoch record the first-hit time at
        # sub-epoch resolution, the same on every row of the run; older files
        # fall back to counting whole epochs from the per-epoch curve.
        out[f"ep_to_{int(t * 100)}"] = (final.get(f"epochs_to_{int(t * 100)}")
                                        if fine else epochs_to(curve, t))
    for k in extra:
        if k in ("upd_survive", "grad_survive", "grad_cos"):
            # whole-run averages; these are per-step rates, not endpoints
            vals = [r[k] for r in curve if r.get(k) is not None]
            out[k] = sum(vals) / len(vals) if vals else None
        else:
            out[k] = final.get(k)
    return out


def med(rs, key):
    """Median over seeds, ignoring runs where the metric is missing."""
    vals = [r[key] for r in rs if r.get(key) is not None]
    return st.median(vals) if vals else None


def cell(v, fmt="{:.4f}", width=10):
    return ("-" if v is None else fmt.format(v)).rjust(width)


def main(IN, OUT):
    runs, extra, fine = load(IN)
    # The sweep records --evals-per-epoch in its metadata. New CSVs always carry
    # first-hit columns, but at the default of 1 they are whole epochs, so the
    # resolution has to come from there rather than from the columns existing.
    src = IN.with_suffix(".meta.json")
    src_meta = json.loads(src.read_text(encoding="utf-8")) if src.exists() else None
    k = ((src_meta or {}).get("args") or {}).get("evals_per_epoch")
    sub_epoch = fine and (k is None or k > 1)
    per_config = defaultdict(list)
    for (fmt, tgt, bits, seed), curve in runs.items():
        per_config[(fmt, tgt, bits)].append(per_run(curve, extra, fine))

    rows = []
    for (fmt, tgt, bits), rs in sorted(per_config.items()):
        row = {
            "format": fmt, "target": tgt, "bits": bits, "seeds": len(rs),
            "final_acc": round(med(rs, "final_acc"), 4),
            "final_acc_min": round(min(r["final_acc"] for r in rs), 4),
            "final_acc_max": round(max(r["final_acc"] for r in rs), 4),
            "best_acc": round(med(rs, "best_acc"), 4),
            "best_epoch": int(med(rs, "best_epoch")),
            "final_loss": round(med(rs, "final_loss"), 4),
            "min_loss": round(med(rs, "min_loss"), 4),
        }
        for t in TARGET_ACCS:
            k = f"ep_to_{int(t * 100)}"
            row[k] = med(rs, k)
            # How many seeds ever got there, so a median epoch count is not
            # mistaken for a converged one.
            row[k + "_n"] = sum(r[k] is not None for r in rs)
        for k in extra:
            v = med(rs, k)
            row[k] = None if v is None else round(v, 6)
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # A summary is only as good as the curves it collapsed, so carry the input
    # file's own metadata forward. If the source was produced at a different
    # commit than this summary, that is visible here instead of inferred from
    # file timestamps.
    save_metadata(OUT, describe_run(
        config={"TARGETS": TARGETS, "EXTRA": EXTRA, "TARGET_ACCS": list(TARGET_ACCS)},
        extra={"status": "complete", "rows": len(rows),
               "source": _repo_relative(IN),
               "source_metadata": src_meta}))

    index = {(r["format"], r["target"], r["bits"]): r for r in rows}
    formats = sorted({r["format"] for r in rows})
    bits = sorted({r["bits"] for r in rows}, reverse=True)
    present = [t for t in TARGETS if any(r["target"] == t for r in rows)]

    def table(title, key, fmt="{:.4f}", note=""):
        print(f"\n--- {title} ---")
        if note:
            print(f"    {note}")
        head = " | ".join(f"{f + ' ' + t:>22}" for f in formats for t in present)
        print(f"{'bits':>4} | {head}")
        for b in bits:
            cells = []
            for f in formats:
                for t in present:
                    r = index.get((f, t, b))
                    cells.append(cell(r[key] if r else None, fmt, 22))
            print(f"{b:>4} | " + " | ".join(cells))

    # 1. Accuracy, with the seed range, because the range is the story wherever
    #    the medians sit within noise of each other.
    for fmt in formats:
        print(f"\n=== {fmt}: final accuracy (min-max over seeds) ===")
        print(f"{'bits':>4} | " + " | ".join(f"{t:>26}" for t in present))
        for b in bits:
            cells = []
            for t in present:
                r = index.get((fmt, t, b))
                cells.append("-".rjust(26) if r is None else
                             (f"{r['final_acc']:.4f} "
                              f"({r['final_acc_min']:.4f}-{r['final_acc_max']:.4f})"
                              ).rjust(26))
            print(f"{b:>4} | " + " | ".join(cells))

    table("final val cross-entropy", "final_loss")

    if "kl" in extra:
        table("KL from the same-seed FP32 model", "kl", "{:.6f}",
              "zero at FP32; the 23-bit row is the cuDNN noise floor, so "
              "believe nothing smaller")
        table("prediction disagreement with the same-seed FP32 model",
              "disagree", "{:.4f}",
              "counts right-to-wrong and wrong-to-right flips, which accuracy "
              "cancels against each other")
    if "logit_rel_err_c" in extra:
        table("relative logit error, row mean removed", "logit_rel_err_c",
              "{:.4f}", "the part of the logit error softmax can actually see")
    if "upd_survive" in extra:
        table("fraction of quantized weights that moved per step",
              "upd_survive", "{:.4f}",
              "without a master a stalled update is discarded; with one it is "
              "only deferred")
    if "grad_survive" in extra:
        table("fraction of nonzero gradient elements surviving quantization",
              "grad_survive", "{:.4f}",
              "1.0 where gradients are not quantized. Elementwise cannot "
              "annihilate a gradient at all; only a shared exponent can.")
        table("cosine similarity of the quantized gradient to FP32",
              "grad_cos", "{:.4f}",
              "a SINGLE-step measure, so it necessarily ranks grad_sr below "
              "grad: stochastic rounding buys unbiasedness with variance. "
              "Judge the two on accuracy and KL, not on this.")

    # 2. Time to target. Low precision slows convergence rather than capping
    #    it, and an endpoint metric cannot show that.
    print("\n--- epochs to reach a target accuracy ---")
    print("    the x/y count is how many seeds ever got there; a median over "
          "fewer than all seeds is censored, not slow.")
    print("    'never' across a row is a ceiling; a rising epoch count with no "
          "'never' is only a slowdown.")
    print("    resolution: " + (f"1/{k} epoch (--evals-per-epoch {k}); fractional "
                                f"epochs, median over seeds" if sub_epoch and k else
                                "sub-epoch; fractional epochs, median over seeds"
                                if sub_epoch else
                                "whole epochs only. Rerun the sweep with "
                                "--evals-per-epoch 10 to resolve differences "
                                "smaller than one epoch."))
    for t in TARGET_ACCS:
        key = f"ep_to_{int(t * 100)}"
        print(f"\n  target {t:.0%}")
        print(f"{'bits':>4} | " + " | ".join(f"{f + ' ' + tg:>22}"
                                             for f in formats for tg in present))
        for b in bits:
            cells = []
            for f in formats:
                for tg in present:
                    r = index.get((f, tg, b))
                    if r is None:
                        cells.append("-".rjust(22))
                    elif r[key + "_n"] == 0:
                        cells.append(f"never (0/{r['seeds']})".rjust(22))
                    else:
                        cells.append(((f"{r[key]:.2f} " if sub_epoch else f"{r[key]:.0f} ")
                                      + f"({r[key + '_n']}/{r['seeds']})"
                                      ).rjust(22))
            print(f"{b:>4} | " + " | ".join(cells))

    # 3. Scaling laws. Accuracy saturates and cannot show one; KL and the
    #    logit error can. Halving the mantissa doubles representation error,
    #    so rel_err_c should go as ~2x per bit removed and KL, being quadratic
    #    in the perturbation, as ~4x.
    if "kl" in extra:
        print("\n--- ratio per bit removed (expect ~2x rel_err_c, ~4x kl) ---")
        print("    normalized by the bit gap, since the bit list is not "
              "evenly spaced")
        asc = sorted(b for b in bits if b < 23)
        for f in formats:
            for t in present:
                line = []
                for lo, hi in zip(asc, asc[1:]):
                    a, c = index.get((f, t, hi)), index.get((f, t, lo))
                    gap = hi - lo
                    if not a or not c or not a.get("kl") or not c.get("kl"):
                        line.append(f"{hi}->{lo}: -")
                        continue
                    rk = (c["kl"] / a["kl"]) ** (1 / gap)
                    rr = ((c["logit_rel_err_c"] / a["logit_rel_err_c"]) ** (1 / gap)
                          if a.get("logit_rel_err_c") else float("nan"))
                    line.append(f"{hi}->{lo}: kl {rk:5.2f} err {rr:4.2f}")
                print(f"{f:11s} {t:13s} " + "  ".join(line))

    # A shared exponent can only lose information relative to per-element ones,
    # so BFP beating elementwise at matched width means a bug, not a finding.
    if {"bfp16", "elementwise"} <= set(formats):
        print("\n--- sanity: bfp16 must not beat elementwise ---")
        bad = []
        for t in present:
            for b in bits:
                e, q = index.get(("elementwise", t, b)), index.get(("bfp16", t, b))
                if e and q and q["final_acc"] - e["final_acc"] > 0.003:
                    bad.append((t, b, e["final_acc"], q["final_acc"]))
        for t, b, e, q in bad:
            print(f"  VIOLATION {t} {b}b: elementwise {e:.4f}, bfp16 {q:.4f}")
        print("  none" if not bad else "")

    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Collapse per-epoch sweep curves into per-configuration "
                    "summaries and print the tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=pathlib.Path,
                   default=DATA / "mnist_cnn_curves.csv",
                   help="curves CSV to summarize")
    p.add_argument("--output", type=pathlib.Path, default=None,
                   help="summary CSV to write. Defaults to the input name "
                        "with _curves replaced by _summary, so a --only sweep "
                        "summarizes to its own file instead of clobbering the "
                        "canonical one.")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"no such curves file: {args.input}")
    out = args.output or args.input.with_name(
        args.input.name.replace("_curves", "_summary")
        if "_curves" in args.input.name else args.input.stem + "_summary.csv")
    main(args.input, out)