import torch, torch.nn as nn
from fpbench.quantize import round_mantissa, quantize_weights
from fpbench.run_metadata import describe_run, save_metadata
from fpbench.cli import (add_sweep_args, guard_output, print_plan,
                         resolve_out, select_conditions)
import argparse, csv, pathlib

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Named so the run metadata file can record them. Previously these were literals
# scattered through the file, which meant the protocol existed only in the
# source at whatever commit happened to produce a given CSV.
SEEDS = 10
EPOCHS = 2000
LR = 1e-2
N_SAMPLES, N_FEATURES = 2048, 16
HIDDEN = 32
BITS = (1, 2, 3, 4, 5, 7, 10, 23)

# (tag, quant_input, quant_weight). This sweep has no weight_master condition,
# so it cannot separate representation error from update-vanishing; the CNN
# sweep does that and finds the gap is almost entirely the latter.
CONDITIONS = [
    ("input", True, False),
    ("weight", False, True),
    ("both", True, True),
]

def run(bits, seed=0, epochs=EPOCHS, quant_input=True, quant_weight=False):
    torch.manual_seed(seed)
    X = torch.randn(N_SAMPLES, N_FEATURES, device=DEVICE)
    y = X @ torch.randn(N_FEATURES, 1, device=DEVICE)  #expected end result, @ is matrix mult
    Xq = round_mantissa(X, bits) if (quant_input and bits < 23) else X

    model = nn.Sequential(nn.Linear(N_FEATURES, HIDDEN), nn.ReLU(),
                          nn.Linear(HIDDEN, 1)).to(DEVICE)
    #nn.Sequential chains layers together
    #nn.Linear(16, 32) - layer that learns, widens 16 numbers to 32
    #ReLU replaces negative nums with 0 - stops two linear layers collapsing into one
    #nn.Linear(32, 1) - collapses 32 numbers back down to one, compared against y

    opt = torch.optim.SGD(model.parameters(), lr=LR)

    if quant_weight:
        quantize_weights(model, bits)   #round the starting weights

    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(Xq), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if quant_weight:
            quantize_weights(model, bits)   #re-round after every update
    return loss.item()

ROOT = pathlib.Path(__file__).resolve().parents[1]


def predict_zero(seed):
    """Loss of a model that outputs 0 for every input, and the target's std."""
    torch.manual_seed(seed)
    X = torch.randn(N_SAMPLES, N_FEATURES, device=DEVICE)
    y = X @ torch.randn(N_FEATURES, 1, device=DEVICE)
    return (y ** 2).mean().item(), y.std().item()


def sweep(args):
    conditions = select_conditions(CONDITIONS, args.only)
    out = resolve_out(args, ROOT / "results" / "data",
                      "train_at_vary_precision.csv", args.only)

    if args.dry_run:
        print_plan(model="MLP 16-32-1 on synthetic regression",
                   conditions=conditions, formats=[("elementwise", None)],
                   bits=args.bits, seeds=args.seeds,
                   budget=f"{args.epochs} full-batch epochs", out=out)
        return

    guard_output(out, args.force)
    rows = []

    # At 23 bits round_mantissa is a no-op, so the baseline is the same for all
    # three conditions. Compute it once per seed instead of once per cell.
    baseline = {s: run(23, seed=s, epochs=args.epochs) for s in range(args.seeds)}
    print("FP32 baselines:", {s: round(v, 5) for s, v in baseline.items()})

    pzero = {s: predict_zero(s)[0] for s in range(args.seeds)}
    for s in range(args.seeds):
        print(f"seed {s}: predict-zero loss {pzero[s]:.2f}, "
              f"target std {predict_zero(s)[1]:.2f}")

    for tag, qi, qw in conditions:
        print(f"\n{tag}")
        for bits in args.bits:
            ratios = []
            for s in range(args.seeds):
                base = baseline[s]
                loss = run(bits, seed=s, quant_input=qi, quant_weight=qw,
                           epochs=args.epochs)
                r = loss / base
                ratios.append(r)
                rows.append({"bits": bits, "target": tag, "seed": s,
                             "loss": loss, "baseline": base, "ratio": r,
                             "predict_zero": pzero[s],
                             "r2": 1 - loss / pzero[s]})
            mean = sum(ratios) / len(ratios)
            print(f"{bits:2d} bits -> {mean:.4f}x baseline  "
                  f"(spread {max(ratios)-min(ratios):.3f})")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bits", "target", "seed", "loss",
                                          "baseline", "ratio", "predict_zero",
                                          "r2"])
        w.writeheader()
        w.writerows(rows)

    # The CSV is written once, at the end, so the metadata follows it rather
    # than bracketing the sweep the way the CNN and transformer ones do.
    save_metadata(out, describe_run(
        config={"SEEDS": args.seeds, "EPOCHS": args.epochs, "LR": LR,
                "BITS": list(args.bits), "N_SAMPLES": N_SAMPLES,
                "N_FEATURES": N_FEATURES, "HIDDEN": HIDDEN,
                "model": "MLP 16-32-1", "optimizer": "SGD",
                "batching": "full-batch",
                "dataset": "synthetic y = X @ w, no noise term",
                "targets": [c[0] for c in conditions]},
        args=args, extra={"status": "complete", "rows": len(rows)}))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precision sweep on a 2-layer MLP, synthetic regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # No --formats or --block: this sweep only ever calls round_mantissa, so
    # offering a BFP flag it silently ignores would be worse than omitting it.
    add_sweep_args(p, conditions=CONDITIONS, bits=list(BITS), seeds=SEEDS,
                   formats=False)
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="full-batch epochs per run")
    sweep(p.parse_args())
