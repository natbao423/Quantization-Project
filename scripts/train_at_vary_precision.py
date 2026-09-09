import torch, torch.nn as nn
from fpbench.quantize import (quantizable_weights, quantize_grads,
                              quantize_weights, round_bfp, round_mantissa)
from fpbench.run_metadata import describe_run, save_metadata
from fpbench.cli import (Progress, add_sweep_args, guard_output, print_plan,
                         resolve_out, select_conditions, select_formats)
import argparse, csv, pathlib
from typing import NamedTuple

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

class Condition(NamedTuple):
    tag: str
    quant_input: bool = False
    quant_weight: bool = False
    quant_grad: bool = False
    grad_stochastic: bool = False


# No weight_master condition, so this sweep cannot separate representation
# error from update-vanishing on the forward path; the CNN sweep does that.
#
# The grad pair is here for a different reason. On the minibatch CNN, gradient
# quantization cost nothing even with 60% of gradient elements annihilated,
# and stochastic rounding made no difference. The proposed explanation is that
# update-vanishing needs PERSISTENT state: a weight is re-rounded every step so
# a discarded update is gone for good, while a CNN gradient is recomputed from
# a fresh minibatch each step, so minibatch noise already dithers it.
#
# This model is FULL-BATCH. The gradient is a deterministic function of fixed
# data and slowly-moving weights, so a component that is persistently small
# relative to its block max is annihilated on every single step. That restores
# the persistence the CNN lacks. If the explanation holds, gradient
# quantization should bite here where it did not there, and grad_sr should
# open a gap over grad where it opened none on the CNN.
CONDITIONS = [
    Condition("input", quant_input=True),
    Condition("weight", quant_weight=True),
    Condition("both", quant_input=True, quant_weight=True),
    Condition("grad", quant_grad=True),
    Condition("grad_sr", quant_grad=True, grad_stochastic=True),
]


def quantize(x, bits, block):
    """block=None gives per-element exponents; block=N gives BFP."""
    if bits >= 23:
        return x
    return round_mantissa(x, bits) if block is None else round_bfp(x, bits, block)


def run(bits, seed=0, epochs=EPOCHS, quant_input=False, quant_weight=False,
        quant_grad=False, grad_stochastic=False, block=None):
    """One full-batch training run. Returns (final_loss, grad_survive).

    Every switch defaults to False, so `run(bits)` quantizes NOTHING and any
    condition has to be asked for. quant_input used to default to True, left
    that way when this function grew its other switches because the sweep
    always passes it explicitly. An ad-hoc probe that did not then silently
    quantized inputs alongside gradients, and the input damage was very nearly
    reported as a gradient result.

    grad_survive is the fraction of nonzero gradient elements still nonzero
    after quantization, averaged over steps. 1.0 when gradients are not
    quantized, and 1.0 for elementwise at any width, since per-element
    exponents mean no gradient can round to zero.
    """
    torch.manual_seed(seed)
    X = torch.randn(N_SAMPLES, N_FEATURES, device=DEVICE)
    y = X @ torch.randn(N_FEATURES, 1, device=DEVICE)  #expected end result, @ is matrix mult
    Xq = quantize(X, bits, block) if quant_input else X

    model = nn.Sequential(nn.Linear(N_FEATURES, HIDDEN), nn.ReLU(),
                          nn.Linear(HIDDEN, 1)).to(DEVICE)
    #nn.Sequential chains layers together
    #nn.Linear(16, 32) - layer that learns, widens 16 numbers to 32
    #ReLU replaces negative nums with 0 - stops two linear layers collapsing into one
    #nn.Linear(32, 1) - collapses 32 numbers back down to one, compared against y

    opt = torch.optim.SGD(model.parameters(), lr=LR)

    if quant_weight:
        quantize_weights(model, bits, block)   #round the starting weights

    track_grads = quant_grad and bits < 23
    g_live = g_kept = 0

    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(Xq), y)
        opt.zero_grad(); loss.backward()

        if track_grads:
            before = torch.cat([w.grad.detach().reshape(-1).clone()
                                for w in quantizable_weights(model)
                                if w.grad is not None])
        if quant_grad:
            quantize_grads(model, bits, block, stochastic=grad_stochastic)
        if track_grads:
            after = torch.cat([w.grad.detach().reshape(-1)
                               for w in quantizable_weights(model)
                               if w.grad is not None])
            live = before != 0
            g_live += live.sum()
            g_kept += (live & (after != 0)).sum()

        opt.step()
        if quant_weight:
            quantize_weights(model, bits, block)   #re-round after every update

    survive = (g_kept.item() / g_live.item()) if track_grads else 1.0
    return loss.item(), survive

ROOT = pathlib.Path(__file__).resolve().parents[1]


def predict_zero(seed):
    """Loss of a model that outputs 0 for every input, and the target's std."""
    torch.manual_seed(seed)
    X = torch.randn(N_SAMPLES, N_FEATURES, device=DEVICE)
    y = X @ torch.randn(N_FEATURES, 1, device=DEVICE)
    return (y ** 2).mean().item(), y.std().item()


FIELDS = ["format", "block", "bits", "target", "seed", "loss", "baseline",
          "ratio", "predict_zero", "r2", "grad_survive"]


def write_rows(out, rows):
    """Rewrite the whole CSV. Called after every cell, not just at the end.

    This sweep used to write once on completion, which made it the only one
    with no way to see progress: no partial CSV, no metadata until it finished,
    and stdout possibly sitting in a pipe buffer. A 34-minute run with no
    observable state is indistinguishable from a hang.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def sweep(args):
    conditions = select_conditions(CONDITIONS, args.only)
    formats = select_formats(args)
    out = resolve_out(args, ROOT / "results" / "data",
                      "train_at_vary_precision.csv", args.only)

    if args.dry_run:
        print_plan(model="MLP 16-32-1 on synthetic regression",
                   conditions=conditions, formats=formats,
                   bits=args.bits, seeds=args.seeds,
                   budget=f"{args.epochs} full-batch epochs", out=out,
                   # measured on a 5070 Ti: 3.5s plain, 4.9s with
                   # gradient tracking. This model is tiny enough that
                   # each step is kernel-launch bound, not compute
                   # bound, so the GPU sits near idle and the wall
                   # clock is set by Python overhead x 2000 epochs.
                   seconds_per_run=4.2)
        return

    guard_output(out, args.force)
    rows = []

    # At 23 bits every quantizer this sweep uses is a no-op, so the baseline is
    # shared by every condition AND every format. Computed once per seed rather
    # than once per cell.
    print(f"building {args.seeds} FP32 baselines", flush=True)
    baseline = {s: run(23, seed=s, epochs=args.epochs)[0]
                for s in range(args.seeds)}
    print("FP32 baselines:", {s: round(v, 5) for s, v in baseline.items()},
          flush=True)

    pzero = {s: predict_zero(s)[0] for s in range(args.seeds)}
    for s in range(args.seeds):
        print(f"seed {s}: predict-zero loss {pzero[s]:.2f}, "
              f"target std {predict_zero(s)[1]:.2f}", flush=True)

    # One step per (format, condition, bit width): the seed loop is inside, and
    # its spread is what the printed line reports.
    bar = Progress(len(formats) * len(conditions) * len(args.bits))

    for fmt_name, block in formats:
        for cond in conditions:
            for bits in args.bits:
                ratios, survives, r2s = [], [], []
                for s in range(args.seeds):
                    base = baseline[s]
                    loss, survive = run(bits, seed=s,
                                        quant_input=cond.quant_input,
                                        quant_weight=cond.quant_weight,
                                        quant_grad=cond.quant_grad,
                                        grad_stochastic=cond.grad_stochastic,
                                        block=block, epochs=args.epochs)
                    r2 = 1 - loss / pzero[s]
                    ratios.append(loss / base)
                    survives.append(survive)
                    r2s.append(r2)
                    rows.append({"format": fmt_name, "block": block or 1,
                                 "bits": bits, "target": cond.tag, "seed": s,
                                 "loss": loss, "baseline": base,
                                 "ratio": loss / base,
                                 "predict_zero": pzero[s], "r2": r2,
                                 "grad_survive": survive})
                mid = sorted(r2s)[len(r2s) // 2]
                bar.step(f"{fmt_name:11s} {cond.tag:8s} {bits:2d}b -> "
                         f"{sum(ratios)/len(ratios):9.3f}x  r2 {mid:7.4f}  "
                         f"gsurv {sum(survives)/len(survives):.4f}  "
                         f"(spread {max(ratios)-min(ratios):.3f})")
                # After the step, so the recorded progress matches the rows
                # actually on disk rather than trailing them by one cell.
                write_rows(out, rows)
                save_metadata(out, describe_run(
                    config={"model": "MLP 16-32-1"}, args=args,
                    extra={"status": "running", "rows": len(rows),
                           "progress": bar.state()}))

    write_rows(out, rows)
    save_metadata(out, describe_run(
        config={"SEEDS": args.seeds, "EPOCHS": args.epochs, "LR": LR,
                "BITS": list(args.bits), "N_SAMPLES": N_SAMPLES,
                "N_FEATURES": N_FEATURES, "HIDDEN": HIDDEN,
                "model": "MLP 16-32-1", "optimizer": "SGD",
                "batching": "full-batch",
                "dataset": "synthetic y = X @ w, no noise term",
                "targets": [c.tag for c in conditions],
                "formats": [f[0] for f in formats]},
        args=args, extra={"status": "complete", "rows": len(rows),
                          "progress": bar.state()}))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precision sweep on a 2-layer MLP, synthetic regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_sweep_args(p, conditions=CONDITIONS, bits=list(BITS), seeds=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="full-batch epochs per run")
    sweep(p.parse_args())
