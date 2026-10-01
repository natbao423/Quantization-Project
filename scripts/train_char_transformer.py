"""Precision sweep on Tiny Shakespeare using a small CharTransformer.
Evaluates activation, weight and gradient quantization on language modeling.

Mechanism columns, recorded per evaluation interval (the EVAL_EVERY steps since
the previous evaluation), with the same meaning as in train_mnist_cnn.py:

    upd_survive     fraction of quantized weight elements whose value changed
                    per optimizer step. Without a master (`weight`) an element
                    that did not move had its update DISCARDED; with one
                    (`weight_master`) it was only DEFERRED into the FP32 copy.
    grad_survive    fraction of nonzero weight-gradient elements still nonzero
                    after quantization. 1.0 elementwise by construction.
    grad_cos        cosine similarity of the quantized gradient to the FP32
                    one. Never use it to rank grad against grad_sr.

All three read 1.0 where nothing of that kind is quantized.

Reference metrics, at every evaluation, against the final model of an FP32 run
trained at the SAME SEED (see fpbench.metrics): kl, disagree, logit_rel_err and
logit_rel_err_c, one row per predicted character. Perplexity saturates and
mixes the perturbation with the solution; these measure the perturbation alone.
The FP32 reference is a separate run, so the 23-bit rows are not exactly zero:
they are the noise floor, and nothing smaller should be believed.

Evaluation covers the whole validation split, tiled into non-overlapping
BLOCK_SIZE windows, so every run scores exactly the same characters in the same
order and the reference metrics can pair them one to one. (It used to be 40
random batches with a fixed seed: consistent between runs, but a sample, with
windows overlapping and some characters scored several times.)

AdamW changes what the grad pair tests, relative to the SGD models. Adam
divides each gradient by a running RMS of its own history, so a component BFP
shrinks is partly scaled back up, and one it zeroes still decays the moments.
The null result on the CNN and MLP is not guaranteed to carry over.
"""

import argparse
import contextlib
import math
import pathlib
import tempfile
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

from fpbench.quantize import (GradSurvival, QuantizedForward, quantize_weights,
                              weight_snapshot)
from fpbench.activations import QuantizedActivations, ActivationStats
from fpbench.metrics import logit_metrics
from fpbench.run_metadata import record_run, save_metadata
from fpbench.cli import (Phase, Progress, add_sweep_args, guard_output,
                         print_plan, resolve_bits, resolve_out,
                         resolve_seeds, select_conditions, select_formats,
                         write_csv)
from fpbench.parallel import DevicePool, gpu_count, resolve_devices, set_current

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tinyshakespeare.txt"
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
       "master/data/tinyshakespeare/input.txt")

# Model configuration
N_LAYER, N_HEAD, D_MODEL, BLOCK_SIZE = 4, 4, 128, 128

# Training configuration
STEPS = 4000
BATCH = 64
LR = 1e-3
EVAL_EVERY = 250
EVAL_BATCH = 128      # windows per evaluation forward pass; memory only
VAL_FRACTION = 0.1


class Condition(NamedTuple):
    """One cell of the design. Named rather than positional, as on the CNN,
    because a misplaced boolean would silently relabel a run."""
    tag: str
    quant_act: bool = False
    quant_weight: bool = False
    master: bool = False            # keep an FP32 master copy of the weights
    quant_grad: bool = False
    grad_stochastic: bool = False   # unbiased dither instead of round-to-nearest


# The same design as the CNN, minus `input` and `both`: the input is token ids,
# which have no mantissa, so there is nothing to round before the embedding.
#
# `weight_master` is the control that was missing. `weight` re-rounds the
# stored weights after every step and so discards any update below half a grid
# step; `weight_master` keeps an FP32 master, so it isolates representation
# error. `activation` is its like-for-like partner. The CNN found those two
# agree within 2x while `weight` sits 40-90x above; this pair tests that here.
#
# `grad`/`grad_sr` round weight gradients between backward and the step, to
# nearest and with an unbiased dither.
#
# The third condition was called "both" until it was renamed to match the CNN,
# where "both" means input + weight and activation + weight is "act_weight".
# The committed char_transformer_curves.csv predates the rename and still says
# "both"; it is the same condition. The first three keep their old order, so a
# sweep of just those still lays out rows as that file does.
CONDITIONS = [
    Condition("activation", quant_act=True),
    Condition("weight", quant_weight=True),
    Condition("act_weight", quant_act=True, quant_weight=True),
    Condition("weight_master", quant_weight=True, master=True),
    Condition("grad", quant_grad=True),
    Condition("grad_sr", quant_grad=True, grad_stochastic=True),
]


def protocol():
    """The constants a CSV cannot show, for the run metadata file."""
    return {
        "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "D_MODEL": D_MODEL,
        "BLOCK_SIZE": BLOCK_SIZE, "STEPS": STEPS, "BATCH": BATCH, "LR": LR,
        "EVAL_EVERY": EVAL_EVERY, "EVAL_BATCH": EVAL_BATCH,
        "VAL_FRACTION": VAL_FRACTION,
        "eval_set": "whole validation split, non-overlapping BLOCK_SIZE windows",
        "reference": "final FP32 model, same seed",
        "model": "CharTransformer", "optimizer": "AdamW",
        "dataset": "tinyshakespeare", "dataset_url": URL,
        # Unlike the CNN sweep, activation hooks here use the library default
        # (Linear and LayerNorm), which includes the output head. Recorded so
        # the two models' activation columns are not compared without noticing.
        "act_hook_types": ["Linear", "LayerNorm"],
    }


class Block(nn.Module):
    """Transformer block with multi-head attention and feed-forward layers."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.fc2 = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        shape = (B, T, N_HEAD, C // N_HEAD)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class CharTransformer(nn.Module):
    """Character-level Transformer (~818k parameters)."""
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def get_data():
    """Downloads and splits the dataset, returning (train_ids, val_ids, vocab_size)."""
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, DATA)

    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=DEVICE)

    # Split sequentially to prevent context leakage across train/val boundaries
    n = int(len(ids) * (1 - VAL_FRACTION))
    return ids[:n], ids[n:], len(chars)


def get_batch(ids, batch_size, generator=None):
    """Yields a batch of inputs and targets from the text sequence."""
    i = torch.randint(len(ids) - BLOCK_SIZE - 1, (batch_size,),
                      generator=generator).to(DEVICE)
    off = torch.arange(BLOCK_SIZE, device=DEVICE)
    idx = i[:, None] + off[None, :]
    return ids[idx], ids[idx + 1]


def val_windows(ids):
    """The validation split as fixed (x, y) windows: (W, BLOCK_SIZE) each.

    Non-overlapping and in order, so every character after the first is
    predicted exactly once, by every run, in the same position. The final
    partial window is dropped.
    """
    n = (len(ids) - 1) // BLOCK_SIZE
    x = ids[:n * BLOCK_SIZE].view(n, BLOCK_SIZE)
    y = ids[1:n * BLOCK_SIZE + 1].view(n, BLOCK_SIZE)
    return x, y


@torch.no_grad()
def collect_logits(model, val, bits, block, quant_act):
    """Every logit on the validation windows, (characters, vocab), in order.

    Activations are rounded here as during training: a run that trains in low
    precision but evaluates in FP32 is measuring something else.
    """
    model.eval()
    x, _ = val
    out = []
    for i in range(0, len(x), EVAL_BATCH):
        with QuantizedActivations(model, bits if quant_act else 23, block):
            out.append(model(x[i:i + EVAL_BATCH]).flatten(0, 1))
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, val, bits, block, quant_act, ref=None):
    """Validation loss and perplexity, plus the reference metrics when `ref`
    (FP32 logits for this seed, from collect_logits) is given."""
    logits = collect_logits(model, val, bits, block, quant_act)
    loss = F.cross_entropy(logits, val[1].flatten()).item()
    out = {"val_loss": loss, "val_ppl": math.exp(loss)}
    if ref is not None:
        out.update(logit_metrics(logits, ref))
    return out


def run(bits, seed, train_ids, val, vocab, block=None, quant_act=False,
        quant_weight=False, master=False, quant_grad=False,
        grad_stochastic=False, steps=STEPS, ref=None, log=None):
    """One training run. Returns (eval curve, model).

    quant_weight with master=False re-rounds the stored weights after every
    optimizer step. With master=True the stored weights stay FP32 and only the
    forward pass, the backward pass and evaluation see quantized values: the
    gradient is taken at the quantized point and applied to the master.

    quant_grad rounds weight gradients in place between backward() and step();
    grad_stochastic uses the unbiased dither, drawn from the global RNG seeded
    below, so those runs stay reproducible.

    Every row carries upd_survive, grad_survive and grad_cos for the steps
    since the previous evaluation (see the module docstring).

    `val` is val_windows(). `ref` is the FP32 reference logits for this seed;
    with None the reference columns are left blank.
    """
    torch.manual_seed(seed)
    model = CharTransformer(vocab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    g = torch.Generator().manual_seed(seed)

    if quant_weight and not master:
        quantize_weights(model, bits, block)      # round the starting weights

    # Only meaningful below 23 bits; at FP32 nothing is quantized to measure.
    track_updates = quant_weight and bits < 23
    track_grads = quant_grad and bits < 23
    act_bits = bits if quant_act else 23

    def weights_seen():
        """The quantized view of the weights under weight_master; else nothing."""
        if quant_weight and master:
            return QuantizedForward(model, bits, block)
        return contextlib.nullcontext()

    moved = elems = 0
    grads = GradSurvival() if track_grads else None
    curve = []
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_ids, BATCH, g)
        # The master is restored on leaving the block; p.grad survives that.
        with weights_seen():
            with QuantizedActivations(model, act_bits, block):
                logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

            opt.zero_grad()
            loss.backward()

        if track_grads:
            grads.quantize(model, bits, block, stochastic=grad_stochastic)

        before = weight_snapshot(model, bits, block, master) if track_updates else None
        opt.step()

        if quant_weight and not master:
            quantize_weights(model, bits, block)   # re-round after each step
        if track_updates:
            after = weight_snapshot(model, bits, block, master)
            moved += (before != after).sum()       # stays on the GPU
            elems += before.numel()

        if step % EVAL_EVERY == 0 or step == steps:
            with weights_seen():
                m = evaluate(model, val, bits, block, quant_act, ref)
            vl, ppl = m["val_loss"], m["val_ppl"]
            # An explicit schema: the CSV header comes from the first row, so
            # every condition must produce the same keys in the same order.
            row = {"step": step, "train_loss": loss.item(),
                   "val_loss": vl, "val_ppl": ppl,
                   "kl": m.get("kl", ""),
                   "disagree": m.get("disagree", ""),
                   "logit_rel_err": m.get("logit_rel_err", ""),
                   "logit_rel_err_c": m.get("logit_rel_err_c", ""),
                   "upd_survive": (moved.item() / elems) if track_updates else 1.0,
                   "grad_survive": grads.survive if track_grads else 1.0,
                   "grad_cos": grads.cos if track_grads else 1.0}
            curve.append(row)
            if log:
                print(f"  step {step:5d}  train {loss.item():.4f}  "
                      f"val {vl:.4f}  ppl {ppl:.3f}"
                      + (f"   kl {m['kl']:.5f}  dis {m['disagree']:.4f}"
                         if ref is not None else "")
                      + (f"   upd {row['upd_survive']:.3f}" if track_updates else "")
                      + (f"   gsurv {row['grad_survive']:.4f} "
                         f"gcos {row['grad_cos']:.4f}" if track_grads else ""))
            moved = elems = 0
            grads = GradSurvival() if track_grads else None
    return curve, model


def diagnose(args):
    """Measures and reports activation outlier statistics before sweeping."""
    train_ids, val_ids, vocab = get_data()
    print(f"vocab {vocab}, uniform-predictor perplexity {vocab}")

    print("\nTraining a short FP32 run to capture representative statistics...")
    _, model = run(23, 0, train_ids, val_windows(val_ids), vocab,
                   steps=args.steps, log=True)

    g = torch.Generator().manual_seed(999)
    with ActivationStats(model, block=16) as stats:
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(get_batch(val_ids, BATCH, g)[0])
                
    rows = stats.print_summary("Per-block exponent statistics, in bits (block=16)")

    worst = max(rows.items(), key=lambda kv: kv[1]["head_p99"])
    print(f"\nWorst module by p99 headroom: {worst[0]} at {worst[1]['head_p99']:.1f} bits")
    print("iid Gaussian reference: headroom p99 = 3.0 bits.")


def smoke(args):
    """Executes a full-precision baseline to verify logic and estimate runtime."""
    train_ids, val_ids, vocab = get_data()
    n = sum(p.numel() for p in CharTransformer(vocab).parameters())
    print(f"Device: {DEVICE}, {n:,} parameters, vocab {vocab}")
    print(f"{len(train_ids):,} train / {len(val_ids):,} val characters")
    print(f"Uniform-predictor perplexity {vocab} (random guessing baseline)")

    t0 = time.time()
    curve, _ = run(23, 0, train_ids, val_windows(val_ids), vocab,
                   steps=args.steps, log=True)
    dt = time.time() - t0

    best = min(curve, key=lambda r: r["val_loss"])
    print(f"\n{dt:.0f}s, {1000*dt/args.steps:.1f} ms/step")
    print(f"Final ppl {curve[-1]['val_ppl']:.3f}  "
          f"Best {best['val_ppl']:.3f} at step {best['step']}")
    print(f"\nSweep estimate: {args.n_configs} configs x {dt/60:.1f} min "
          f"= {args.n_configs * dt / 3600:.1f} h")


# Per-worker state, filled by worker_setup. See fpbench.parallel: with one
# device the "worker" is this process and this is just a cache.
_WORKER = {}


def worker_setup(device):
    """Point this process at `device` and load the text onto it."""
    global DEVICE
    DEVICE = device
    train_ids, val_ids, vocab = get_data()
    _WORKER.update(train=train_ids, val=val_windows(val_ids), vocab=vocab, refs={})


def reference_cell(job):
    """One FP32 reference run. Saves its validation logits to job["path"] and
    returns (val_ppl, seconds).

    Saved to a file rather than returned: about 29 MB of logits per seed would
    otherwise be copied into every one of the sweep's runs on its way to a
    worker. Each worker loads a seed's file once and keeps it.
    """
    t0 = time.time()
    curve, model = run(23, job["seed"], _WORKER["train"], _WORKER["val"],
                       _WORKER["vocab"], steps=job["steps"])
    torch.save(collect_logits(model, _WORKER["val"], 23, None, False).cpu(),
               job["path"])
    return curve[-1]["val_ppl"], time.time() - t0


def load_reference(path):
    refs = _WORKER["refs"]
    if path not in refs:
        refs[path] = torch.load(path, map_location=DEVICE)
    return refs[path]


def sweep_cell(job):
    """One configuration: (eval curve, seconds)."""
    t0 = time.time()
    cond = job["cond"]
    curve, _ = run(job["bits"], job["seed"], _WORKER["train"], _WORKER["val"],
                   _WORKER["vocab"], block=job["block"],
                   quant_act=cond.quant_act,
                   quant_weight=cond.quant_weight,
                   master=cond.master,
                   quant_grad=cond.quant_grad,
                   grad_stochastic=cond.grad_stochastic,
                   steps=job["steps"], ref=load_reference(job["ref"]))
    return curve, time.time() - t0


def sweep(args):
    """Executes the full precision configuration grid, saving metrics to CSV."""
    out = resolve_out(args, ROOT / "results" / "data",
                      "char_transformer_curves.csv", args.only)
    conditions = select_conditions(CONDITIONS, args.only)
    formats = select_formats(args)

    if args.dry_run:
        print_plan(model="CharTransformer on Tiny Shakespeare",
                   conditions=conditions, formats=formats, bits=args.bits,
                   seeds=args.seeds, budget=f"{args.steps} steps", out=out,
                   # Measured on a 5070 Ti at 4000 steps with the other GPU
                   # also busy: ~20s (FP32) to ~70s (BFP weight_master);
                   # 45s is a mix of the six conditions.
                   seconds_per_run=45.0, gpus=gpu_count(args.device_list),
                   extra={"devices": " ".join(args.device_list)})
        return

    guard_output(out, args.force)

    # Here, before any worker starts, so two workers never download at once.
    # Only the vocabulary is kept; each worker loads its own copy.
    with Phase("loading Tiny Shakespeare"):
        _, _, vocab = get_data()

    # Grid order: rows are written in this order whatever order the runs
    # finish in, so a parallel sweep's CSV matches a sequential one.
    grid = [(fmt, block, cond, bits, seed)
            for fmt, block in formats
            for cond in conditions
            for bits in args.bits
            for seed in range(args.seeds)]
    meta = {"conditions": [c[0] for c in conditions],
            "formats": [f[0] for f in formats], "vocab": vocab,
            "n_configs": len(grid)}
    done = {}

    # References live in a temporary directory for the length of the sweep;
    # see reference_cell for why they go through files at all.
    with record_run(out, config=protocol(), args=args, extra=meta) as rec, \
         tempfile.TemporaryDirectory(prefix="fpbench_refs_") as tmp, \
         DevicePool(args.device_list, worker_setup) as pool:
        print(f"building {args.seeds} FP32 references "
              f"(one extra run per seed, needed for the reference metrics)",
              flush=True)
        refs = {s: str(pathlib.Path(tmp) / f"ref_seed{s}.pt")
                for s in range(args.seeds)}
        ref_jobs = [{"seed": s, "steps": args.steps, "path": refs[s]}
                    for s in range(args.seeds)]
        for i, (ppl, secs) in pool.imap(reference_cell, ref_jobs):
            print(f"reference seed{ref_jobs[i]['seed']}: ppl {ppl:.3f}  "
                  f"({secs:.0f}s)", flush=True)

        bar = Progress(len(grid))
        jobs = [{"cond": cond, "block": block, "bits": bits,
                 "seed": seed, "steps": args.steps, "ref": refs[seed]}
                for _, block, cond, bits, seed in grid]
        for i, (curve, secs) in pool.imap(sweep_cell, jobs):
            fmt, block, cond, bits, seed = grid[i]
            done[i] = [{"format": fmt, "block": block or 1, "target": cond.tag,
                        "bits": bits, "seed": seed, "vocab": vocab, **r}
                       for r in curve]
            last = curve[-1]
            bar.step(f"{fmt:11s} {cond.tag:13s} {bits:2d}b seed{seed} -> "
                     f"ppl {last['val_ppl']:8.3f} "
                     f"kl {last['kl']:.5f} "
                     f"upd {last['upd_survive']:.3f} "
                     f"gsurv {last['grad_survive']:.4f} ({secs:.0f}s)")

            # Rewrite continuously to prevent data loss on crash
            rows = [r for j in sorted(done) for r in done[j]]
            write_csv(out, rows)
            rec["rows"] = len(rows)
            rec["progress"] = bar.state()
            save_metadata(out, rec)
    print(f"\nWrote {rec.get('rows', 0)} rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precision sweep on Tiny Shakespeare with a small transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    modes = p.add_argument_group("modes (default: the full precision sweep)")
    modes.add_argument("--smoke", action="store_true",
                       help="one FP32 run; prints curves and a sweep estimate")
    modes.add_argument("--diagnose", action="store_true",
                       help="activation outlier statistics, before sweeping")

    # Default bits match the committed CSV. They previously started at 3, so
    # a bare invocation did not reproduce the sweep this repo ships.
    add_sweep_args(p, conditions=CONDITIONS,
                   bits=[1, 2, 3, 4, 5, 7, 10, 23], seeds=3)

    p.add_argument("--steps", type=int, default=STEPS,
                   help="training budget, frozen across bit widths")

    args = p.parse_args()
    # --smoke and --diagnose train at FP32 only, so they never ask.
    sweeping = not (args.smoke or args.diagnose)
    resolve_bits(p, args, prompt=sweeping)
    per_seed = (len(args.bits) * len(select_conditions(CONDITIONS, args.only))
                * len(select_formats(args)))
    resolve_seeds(p, args, prompt=sweeping, runs_per_seed=per_seed)
    args.n_configs = per_seed * args.seeds

    # Resolved into args so the run metadata records where it actually ran.
    # Only the sweep spreads across devices; --smoke and --diagnose run here.
    args.device_list = resolve_devices(args.devices, args.jobs_per_gpu)
    DEVICE = args.device_list[0]
    set_current(DEVICE)

    if args.diagnose:
        diagnose(args)
    elif args.smoke:
        smoke(args)
    else:
        sweep(args)
