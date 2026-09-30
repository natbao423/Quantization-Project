import torch, torch.nn as nn
from fpbench.quantize import GradSurvival, quantize, quantize_weights
from fpbench.run_metadata import record_run, save_metadata
from fpbench.cli import (Progress, add_sweep_args, guard_output, print_plan,
                         resolve_bits, resolve_out, resolve_seeds,
                         select_conditions, select_formats, write_csv)
from fpbench.parallel import DevicePool, gpu_count, resolve_devices, set_current
import argparse, pathlib, time
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

    # quantize_grads is a no-op at 23 bits, so only below it is there anything
    # to do or measure. No cosine: this model is launch-bound, and the extra
    # kernels would show in the wall clock for a column the sweep never used.
    track_grads = quant_grad and bits < 23
    grads = GradSurvival(track_cos=False) if track_grads else None

    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(Xq), y)
        opt.zero_grad(); loss.backward()

        if track_grads:
            grads.quantize(model, bits, block, stochastic=grad_stochastic)

        opt.step()
        if quant_weight:
            quantize_weights(model, bits, block)   #re-round after every update

    return loss.item(), (grads.survive if track_grads else 1.0)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def predict_zero(seed):
    """Loss of a model that outputs 0 for every input, and the target's std."""
    torch.manual_seed(seed)
    X = torch.randn(N_SAMPLES, N_FEATURES, device=DEVICE)
    y = X @ torch.randn(N_FEATURES, 1, device=DEVICE)
    return (y ** 2).mean().item(), y.std().item()


FIELDS = ["format", "block", "bits", "target", "seed", "loss", "baseline",
          "ratio", "predict_zero", "r2", "grad_survive"]


def worker_setup(device):
    """Point this process at `device`. The data is generated per run, from the
    seed, so there is nothing to load."""
    global DEVICE
    DEVICE = device


def baseline_cell(job):
    """FP32 loss for one seed."""
    return run(23, seed=job["seed"], epochs=job["epochs"])[0]


def sweep_cell(job):
    """Every seed of one (format, condition, width): [(loss, grad_survive)].

    A cell is all the seeds, not one, so the per-cell line can report the seed
    spread as it always has. At about 4s a run that is ~40s a cell, small
    enough to share out evenly.
    """
    t0 = time.time()
    cond = job["cond"]
    out = [run(job["bits"], seed=s,
               quant_input=cond.quant_input,
               quant_weight=cond.quant_weight,
               quant_grad=cond.quant_grad,
               grad_stochastic=cond.grad_stochastic,
               block=job["block"], epochs=job["epochs"])
           for s in range(job["seeds"])]
    return out, time.time() - t0


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
                   seconds_per_run=4.2, gpus=gpu_count(args.device_list),
                   extra={"devices": " ".join(args.device_list)})
        return

    guard_output(out, args.force)

    config = {"SEEDS": args.seeds, "EPOCHS": args.epochs, "LR": LR,
              "BITS": list(args.bits), "N_SAMPLES": N_SAMPLES,
              "N_FEATURES": N_FEATURES, "HIDDEN": HIDDEN,
              "model": "MLP 16-32-1", "optimizer": "SGD",
              "batching": "full-batch",
              "dataset": "synthetic y = X @ w, no noise term",
              "targets": [c.tag for c in conditions],
              "formats": [f[0] for f in formats]}

    # Grid order: rows are written in this order whatever order the cells
    # finish in, so a parallel sweep's CSV matches a sequential one.
    grid = [(fmt_name, block, cond, bits)
            for fmt_name, block in formats
            for cond in conditions
            for bits in args.bits]
    seeds = range(args.seeds)
    done = {}

    with record_run(out, config=config, args=args) as rec,          DevicePool(args.device_list, worker_setup) as pool:
        # At 23 bits every quantizer this sweep uses is a no-op, so the
        # baseline is shared by every condition AND every format. Computed
        # once per seed rather than once per cell.
        print(f"building {args.seeds} FP32 baselines", flush=True)
        baseline = dict(zip(seeds, pool.map(
            baseline_cell, [{"seed": s, "epochs": args.epochs} for s in seeds])))
        print("FP32 baselines:", {s: round(v, 5) for s, v in baseline.items()},
              flush=True)

        pzero = {s: predict_zero(s)[0] for s in seeds}
        for s in seeds:
            print(f"seed {s}: predict-zero loss {pzero[s]:.2f}, "
                  f"target std {predict_zero(s)[1]:.2f}", flush=True)

        # One step per (format, condition, bit width): the seed loop is
        # inside, and its spread is what the printed line reports.
        bar = Progress(len(grid))
        jobs = [{"cond": cond, "block": block, "bits": bits,
                 "seeds": args.seeds, "epochs": args.epochs}
                for _, block, cond, bits in grid]
        for i, (results, secs) in pool.imap(sweep_cell, jobs):
            fmt_name, block, cond, bits = grid[i]
            done[i] = []
            for s, (loss, survive) in zip(seeds, results):
                done[i].append({"format": fmt_name, "block": block or 1,
                                "bits": bits, "target": cond.tag, "seed": s,
                                "loss": loss, "baseline": baseline[s],
                                "ratio": loss / baseline[s],
                                "predict_zero": pzero[s],
                                "r2": 1 - loss / pzero[s],
                                "grad_survive": survive})
            ratios = [r["ratio"] for r in done[i]]
            r2s = sorted(r["r2"] for r in done[i])
            survives = [r["grad_survive"] for r in done[i]]
            bar.step(f"{fmt_name:11s} {cond.tag:8s} {bits:2d}b -> "
                     f"{sum(ratios)/len(ratios):9.3f}x  r2 {r2s[len(r2s) // 2]:7.4f}  "
                     f"gsurv {sum(survives)/len(survives):.4f}  "
                     f"(spread {max(ratios)-min(ratios):.3f}, {secs:.0f}s)")

            # Rewritten after every cell, not just at the end. This sweep used
            # to write once on completion, and a 34-minute run with no
            # observable state is indistinguishable from a hang.
            rows = [r for j in sorted(done) for r in done[j]]
            write_csv(out, rows, FIELDS)
            rec["rows"] = len(rows)
            rec["progress"] = bar.state()
            save_metadata(out, rec)
    print(f"wrote {rec.get('rows', 0)} rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precision sweep on a 2-layer MLP, synthetic regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_sweep_args(p, conditions=CONDITIONS, bits=list(BITS), seeds=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="full-batch epochs per run")
    args = p.parse_args()
    resolve_bits(p, args)
    resolve_seeds(p, args, runs_per_seed=len(args.bits)
                  * len(select_conditions(CONDITIONS, args.only))
                  * len(select_formats(args)))
    # Resolved into args so the run metadata records where it actually ran.
    args.device_list = resolve_devices(args.devices, args.jobs_per_gpu)
    DEVICE = args.device_list[0]
    set_current(DEVICE)
    sweep(args)
