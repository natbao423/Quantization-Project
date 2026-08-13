"""Collapse per-epoch curves into per-configuration summaries.

    python scripts/summarize_curves.py

Reads results/data/mnist_cnn_curves.csv, writes mnist_cnn_summary.csv, and
prints the tables.

Reports final-epoch and best-epoch numbers side by side. Final is the honest
frozen-budget number and is what the headline should use. Best is a check: if
a configuration peaked well before the budget, the final number is measuring
decline rather than precision.
"""

import csv
import pathlib
import statistics as st
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "data" / "mnist_cnn_curves.csv"
OUT = ROOT / "results" / "data" / "mnist_cnn_summary.csv"

TARGETS = ["input", "weight", "both"]


def load():
    """(format, target, bits, seed) -> curve sorted by epoch."""
    runs = defaultdict(list)
    with IN.open() as f:
        for r in csv.DictReader(f):
            key = (r["format"], r["target"], int(r["bits"]), int(r["seed"]))
            runs[key].append((int(r["epoch"]), float(r["val_acc"]),
                              float(r["val_loss"])))
    for v in runs.values():
        v.sort()
    return runs


def per_run(curve):
    final_ep, final_acc, final_loss = curve[-1]
    best_ep, best_acc, _ = max(curve, key=lambda t: t[1])
    return {
        "final_acc": final_acc,
        "final_loss": final_loss,
        "best_acc": best_acc,
        "best_epoch": best_ep,
        "min_loss": min(t[2] for t in curve),
        "epochs": final_ep,
    }


def main():
    runs = load()
    per_config = defaultdict(list)
    for (fmt, tgt, bits, seed), curve in runs.items():
        per_config[(fmt, tgt, bits)].append(per_run(curve))

    rows = []
    for (fmt, tgt, bits), rs in sorted(per_config.items()):
        med = lambda k: st.median(r[k] for r in rs)
        rows.append({
            "format": fmt, "target": tgt, "bits": bits, "seeds": len(rs),
            "final_acc": round(med("final_acc"), 4),
            "final_acc_min": round(min(r["final_acc"] for r in rs), 4),
            "final_acc_max": round(max(r["final_acc"] for r in rs), 4),
            "best_acc": round(med("best_acc"), 4),
            "best_epoch": int(med("best_epoch")),
            "final_loss": round(med("final_loss"), 4),
            "min_loss": round(med("min_loss"), 4),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    index = {(r["format"], r["target"], r["bits"]): r for r in rows}
    formats = sorted({r["format"] for r in rows})
    bits = sorted({r["bits"] for r in rows}, reverse=True)

    for fmt in formats:
        print(f"\n=== {fmt} ===")
        head = " | ".join(f"{t + ' fin/best':>18}" for t in TARGETS)
        print(f"{'bits':>4} | {head} | best ep")
        for b in bits:
            cells, eps = [], []
            for t in TARGETS:
                r = index.get((fmt, t, b))
                if r is None:
                    cells.append(f"{'-':>18}")
                    continue
                cells.append(f"{r['final_acc']:.4f} / {r['best_acc']:.4f}".rjust(18))
                eps.append(r["best_epoch"])
            ep = f"{int(st.median(eps))}" if eps else "-"
            print(f"{b:>4} | " + " | ".join(cells) + f" | {ep:>7}")

    print("\n--- final val loss (resolves where accuracy saturates) ---")
    print(f"{'bits':>4} | " + " | ".join(
        f"{f + ' ' + t:>20}" for f in formats for t in ("input", "weight")))
    for b in bits:
        cells = []
        for f in formats:
            for t in ("input", "weight"):
                r = index.get((f, t, b))
                cells.append(f"{r['final_loss']:.4f}".rjust(20) if r else f"{'-':>20}")
        print(f"{b:>4} | " + " | ".join(cells))

    # A shared exponent can only lose information relative to per-element ones,
    # so BFP beating elementwise at matched width means a bug, not a finding.
    if {"bfp16", "elementwise"} <= set(formats):
        print("\n--- sanity: bfp16 must not beat elementwise ---")
        bad = [(t, b, index[("elementwise", t, b)]["final_acc"],
                index[("bfp16", t, b)]["final_acc"])
               for t in TARGETS for b in bits
               if index[("bfp16", t, b)]["final_acc"]
               - index[("elementwise", t, b)]["final_acc"] > 0.003]
        for t, b, e, f in bad:
            print(f"  VIOLATION {t} {b}b: elementwise {e:.4f}, bfp16 {f:.4f}")
        print("  none" if not bad else "")

    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
