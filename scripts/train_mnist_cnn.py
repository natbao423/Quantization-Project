"""Precision sweep on MNIST with a small CNN.

Adds three things the MLP study could not have:
  - depth, so error has somewhere to compound
  - weight sharing, so one quantized kernel's error applies at every position
  - a bounded metric (top-1 accuracy), so runs are comparable without a ratio

The whole dataset lives on the GPU. At 20k parameters the model is far too
small to hide dataloader latency, so a CPU DataLoader leaves the GPU idle most
of the time.

Modes:
    --smoke         one FP32 run, prints curves and a sweep time estimate
    --stats         activation distribution shape, for the outlier comparison
    --batch-study   does gradient noise smooth out the quantization cliff?
    --test          final test accuracy, once, at the end of the project
    (no flag)       the full precision sweep
"""

import argparse
import csv
import pathlib
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from fpbench.quantize import round_mantissa, round_bfp, quantize_weights
from fpbench.activations import ActivationStats

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = pathlib.Path(__file__).resolve().parents[1]

# The validation split must NOT depend on the run seed, or every configuration
# is scored against a different validation set and nothing is comparable.
SPLIT_SEED = 12345
VAL_SIZE = 5_000

EPOCHS = 12          # from the FP32 curve: val loss bottoms at 10-11, then
                     # climbs. Frozen here for every bit width.
BATCH = 128
LR = 0.1
MOMENTUM = 0.0       # see notes at the bottom of this file
MICRO = 1024         # largest chunk sent to the GPU at once; not a
                     # hyperparameter, only a memory limit

# Layers whose weights are never quantized. Mirrors quantize_weights.
SKIP_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)


class SmallCNN(nn.Module):
    """20,490 parameters. Two conv layers, one linear head, no normalization."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_data(limit_train=None):
    """Return (train, val, test), each a (x, y) tuple already on the GPU."""
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.MNIST(ROOT / "data", train=True, download=True, transform=tf)
    test = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tf)

    def to_gpu(ds, idx=None):
        idx = range(len(ds)) if idx is None else idx
        x = torch.stack([ds[i][0] for i in idx]).to(DEVICE)
        y = torch.tensor([ds[i][1] for i in idx]).to(DEVICE)
        return x, y

    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(full), generator=g).tolist()
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    if limit_train:
        train_idx = train_idx[:limit_train]
    return to_gpu(full, train_idx), to_gpu(full, val_idx), to_gpu(test)


def batches(data, batch_size, generator=None, shuffle=False):
    """Iterate (x, y) slices of a GPU-resident dataset."""
    x, y = data
    # randperm is built on CPU because `generator` is a CPU generator; PyTorch
    # requires the two to be on the same device.
    order = (torch.randperm(len(y), generator=generator).to(DEVICE)
             if shuffle else torch.arange(len(y), device=DEVICE))
    for i in range(0, len(y), batch_size):
        j = order[i:i + batch_size]
        yield x[j], y[j]


def quantize(x, bits, block):
    """block=None gives per-element exponents; block=N gives BFP."""
    if bits >= 23:
        return x
    return round_mantissa(x, bits) if block is None else round_bfp(x, bits, block)


class QuantizedForward:
    """FP32 master weights: params hold quantized values only inside the block.

        with QuantizedForward(model, bits, block):
            loss = criterion(model(x), y)
            loss.backward()
        opt.step()                      # updates the FP32 master

    The gradient is computed at the quantized point but applied to the
    full-precision parameter, so updates smaller than the grid spacing still
    accumulate. This is the arrangement production mixed-precision pipelines
    use, and it is the control for `quantize_weights`, which has no master copy
    and therefore discards any update below half a grid step.

    Separating the two isolates representation error (present in both) from
    update-vanishing (present only without a master).
    """

    def __init__(self, model, bits, block=None):
        self.model, self.bits, self.block = model, bits, block
        self.saved = []

    @torch.no_grad()
    def __enter__(self):
        if self.bits >= 23:
            return self
        for mod in self.model.modules():
            if isinstance(mod, SKIP_TYPES):
                continue
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_floating_point(w):
                continue
            self.saved.append((w, w.detach().clone()))
            w.copy_(quantize(w, self.bits, self.block))
        return self

    @torch.no_grad()
    def __exit__(self, *exc):
        for w, master in self.saved:      # p.grad survives this restore
            w.copy_(master)
        self.saved.clear()
        return False


@torch.no_grad()
def evaluate(model, data, bits, block, quant_input):
    model.eval()
    loss_sum = correct = n = 0
    for x, y in batches(data, 512):
        if quant_input:
            x = quantize(x, bits, block)
        out = model(x)
        loss_sum += nn.functional.cross_entropy(out, y, reduction="sum")
        correct += (out.argmax(1) == y).sum()
        n += y.numel()
    return loss_sum.item() / n, correct.item() / n


def run(bits, seed, train, val, block=None, quant_input=False,
        quant_weight=False, master=False, epochs=EPOCHS, log=None):
    """One training run, budgeted in epochs. Returns (curve, model).

    quant_weight with master=False re-rounds the stored weights after every
    optimizer step. With master=True the stored weights stay FP32 and only the
    forward pass sees quantized values.
    """
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # batch order is tied to the run seed, so it varies the same way weight
    # initialization does
    g = torch.Generator().manual_seed(seed)

    if quant_weight and not master:
        quantize_weights(model, bits, block)      # round the starting weights

    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = n = 0
        for x, y in batches(train, BATCH, g, shuffle=True):
            if quant_input:
                x = quantize(x, bits, block)

            opt.zero_grad()
            if quant_weight and master:
                with QuantizedForward(model, bits, block):
                    out = model(x)
                    loss = nn.functional.cross_entropy(out, y)
                    loss.backward()
            else:
                out = model(x)
                loss = nn.functional.cross_entropy(out, y)
                loss.backward()

            opt.step()
            if quant_weight and not master:
                quantize_weights(model, bits, block)   # re-round after each step

            # accumulated on the GPU; .item() here would sync twice per step
            loss_sum += loss.detach() * y.numel()
            correct += (out.argmax(1) == y).sum()
            n += y.numel()

        if quant_weight and master:
            with QuantizedForward(model, bits, block):
                val_loss, val_acc = evaluate(model, val, bits, block, quant_input)
        else:
            val_loss, val_acc = evaluate(model, val, bits, block, quant_input)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum.item() / n,
            "train_acc": correct.item() / n,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        curve.append(row)
        if log:
            print(f"  epoch {epoch:2d}  train {row['train_loss']:.4f}/"
                  f"{row['train_acc']:.4f}   val {val_loss:.4f}/{val_acc:.4f}")
    return curve, model


def run_budget(bits, seed, train, val, block=None, quant_input=False,
               quant_weight=True, master=False, updates=5160, batch=BATCH,
               eval_every=None, log=None):
    """One training run budgeted in OPTIMIZER UPDATES rather than epochs.

    Batch size and update count are decoupled here so batch size can be varied
    while holding the amount of learning fixed. Batches larger than MICRO are
    split into microbatches whose gradients are accumulated, which is
    mathematically identical to computing the whole batch at once but bounds
    memory: a full 55,000-image forward pass would need roughly 9.5 GB of
    activations.

    LR is deliberately NOT scaled with batch size. The usual advice is to scale
    it, but that would defeat the experiment: holding LR fixed keeps the mean
    update per step constant while the noise around it shrinks as batch grows,
    which isolates gradient noise as the only variable.
    """
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    g = torch.Generator().manual_seed(seed)
    eval_every = eval_every or updates

    if quant_weight and not master:
        quantize_weights(model, bits, block)

    x_all, y_all = train
    n = len(y_all)
    curve, step = [], 0

    while step < updates:
        order = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n - batch + 1, batch):    # drop any short final batch
            idx = order[i:i + batch]
            opt.zero_grad()
            model.train()

            for j in range(0, batch, MICRO):
                sub = idx[j:j + MICRO]
                x, y = x_all[sub], y_all[sub]
                if quant_input:
                    x = quantize(x, bits, block)
                # scale so the accumulated gradient is the mean over the whole
                # logical batch, not a sum of microbatch means
                scale = len(sub) / batch
                if quant_weight and master:
                    with QuantizedForward(model, bits, block):
                        loss = nn.functional.cross_entropy(model(x), y) * scale
                        loss.backward()
                else:
                    loss = nn.functional.cross_entropy(model(x), y) * scale
                    loss.backward()

            opt.step()
            if quant_weight and not master:
                quantize_weights(model, bits, block)
            step += 1

            if step % eval_every == 0 or step == updates:
                if quant_weight and master:
                    with QuantizedForward(model, bits, block):
                        vl, va = evaluate(model, val, bits, block, quant_input)
                else:
                    vl, va = evaluate(model, val, bits, block, quant_input)
                curve.append({"step": step, "val_loss": vl, "val_acc": va})
                if log:
                    print(f"  update {step:5d}  val {vl:.4f}/{va:.4f}")
            if step >= updates:
                break
    return curve, model

def ptq_check(args):
    """Post-training quantization of an FP32 model.

    If a plain FP32 model survives 1-bit-mantissa weights, then MNIST simply
    tolerates this representation and weight_master's flat curve is real.
    If it craters, weight_master should NOT be flat, and there is a bug.
    """
    train, val, _ = get_data()
    _, model = run(23, seed=0, train=train, val=val, epochs=args.epochs)
    base = evaluate(model, val, 23, None, False)
    print(f"FP32                {base[1]:.4f}")

    for bits in [10, 7, 5, 4, 3, 2, 1, 0]:
        with QuantizedForward(model, bits, None):
            w = model.net[0].weight
            nuniq = w.abs().unique().numel()
            loss, acc = evaluate(model, val, 23, None, False)
        print(f"{bits:2d} mantissa bits    {acc:.4f}   "
              f"(conv1 has {nuniq} distinct magnitudes of {w.numel()})")


def smoke(args):
    """One FP32 run. Confirm the curves look sane before spending hours."""
    train, val, _ = get_data(args.limit_train)
    n_params = sum(p.numel() for p in SmallCNN().parameters())
    print(f"device {DEVICE}, {n_params:,} parameters, "
          f"{len(train[1]):,} train / {len(val[1]):,} val")

    t0 = time.time()
    curve, _ = run(23, seed=0, train=train, val=val, epochs=args.epochs, log=True)
    dt = time.time() - t0

    best_acc = max(curve, key=lambda r: r["val_acc"])
    best_loss = min(curve, key=lambda r: r["val_loss"])
    print(f"\n{dt:.0f}s total, {dt/args.epochs:.1f}s/epoch")
    print(f"final val acc {curve[-1]['val_acc']:.4f}  "
          f"best {best_acc['val_acc']:.4f} at epoch {best_acc['epoch']}")
    print(f"val loss bottoms at epoch {best_loss['epoch']} "
          f"({best_loss['val_loss']:.4f}), ends {curve[-1]['val_loss']:.4f}")
    print(f"train-val accuracy gap at end: "
          f"{curve[-1]['train_acc'] - curve[-1]['val_acc']:+.4f}")
    print(f"\nFull sweep estimate: {args.n_configs} configs x {dt/60:.1f} min "
          f"= {args.n_configs * dt / 3600:.1f} h")


def sweep(args):
    """The precision sweep. Four weight/input conditions x two formats."""
    train, val, _ = get_data(args.limit_train)
    rows = []
    tag_suffix = "_" + "_".join(args.only) if args.only else ""
    out = ROOT / "results" / "data" / f"mnist_cnn_curves{tag_suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    # (tag, quant_input, quant_weight, master)
    conditions = [
        ("input", True, False, False),
        ("weight", False, True, False),         # no master: update-vanishing
        ("weight_master", False, True, True),   # FP32 master: representation
                                                # error only
        ("both", True, True, False),
    ]
    if args.only:
        conditions = [c for c in conditions if c[0] in args.only]
    formats = [("elementwise", None), ("bfp16", 16)]

    for fmt_name, block in formats:
        for tag, qi, qw, mw in conditions:
            for bits in args.bits:
                for seed in range(args.seeds):
                    t0 = time.time()
                    curve, _ = run(bits, seed, train, val, block=block,
                                   quant_input=qi, quant_weight=qw, master=mw,
                                   epochs=args.epochs)
                    for r in curve:
                        rows.append({"format": fmt_name, "block": block or 1,
                                     "target": tag, "bits": bits, "seed": seed,
                                     **r})
                    final = curve[-1]["val_acc"]
                    best = max(r["val_acc"] for r in curve)
                    print(f"{fmt_name:11s} {tag:13s} {bits:2d}b seed{seed} -> "
                          f"final {final:.4f} best {best:.4f} "
                          f"({time.time()-t0:.0f}s)")

                    with out.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)      # rewrite each run, so a crash
                                               # does not lose everything
    print(f"\nwrote {len(rows)} rows to {out}")


def batch_study(args):
    """Does gradient noise smooth out the quantization cliff?

    The MLP was full-batch and showed a sharp, initialization-dependent cliff
    at 4 bits, with per-seed results spanning a twenty-fold range. The CNN is
    minibatch and shows no cliff at all. If gradient noise is the reason, the
    cliff should reappear as batch size grows and gradients get less noisy: a
    weight whose mean update sits below half a grid step will still see
    individual updates that cross it, so long as the noise is large enough.
    This is dithering, the same effect that recovers sub-quantum signal in
    audio.

    Update count is held fixed across batch sizes so this measures noise, not
    training length. Watch the SEED SPREAD at 4 bits, not the median: the MLP's
    signature was bimodality across seeds, and that is what should return.
    """
    train, val, _ = get_data()
    rows = []
    out = ROOT / "results" / "data" / "mnist_batch_study.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    for batch in args.batches:
        for bits in args.bits:
            accs = []
            for seed in range(args.seeds):
                t0 = time.time()
                curve, _ = run_budget(bits, seed, train, val,
                                      quant_weight=True, master=args.master,
                                      updates=args.updates, batch=batch)
                r = curve[-1]
                accs.append(r["val_acc"])
                rows.append({"batch": batch, "bits": bits, "seed": seed,
                             "updates": args.updates, "master": args.master,
                             **r})
                print(f"batch {batch:6d}  {bits:2d}b seed{seed} -> "
                      f"acc {r['val_acc']:.4f}  ({time.time()-t0:.0f}s)")
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            print(f"  -> batch {batch} {bits}b spread "
                  f"{max(accs)-min(accs):.4f} over {len(accs)} seeds\n")
    print(f"wrote {len(rows)} rows to {out}")


def stats(args):
    """Activation distribution shape, for comparison with the transformer."""
    train, val, _ = get_data()
    _, model = run(23, seed=0, train=train, val=val, epochs=args.epochs)

    # The CNN has no LayerNorm, so the default module types would match only
    # the final Linear. Conv2d outputs are the activations that matter here.
    # These are pre-ReLU, matching the transformer, whose hooked tensors are
    # also pre-nonlinearity.
    with ActivationStats(model, block=16, types=(nn.Conv2d, nn.Linear)) as s:
        model.eval()
        with torch.no_grad():
            for x, _ in batches(val, 512):
                model(x)
    s.print_summary("MNIST CNN, per-block exponent statistics (block=16)")

    # Post-ReLU as well: roughly half the elements are exactly zero, which is
    # where the zero handling in block_exponent_stats earns its keep.
    with ActivationStats(model, block=16, types=(nn.ReLU,)) as s:
        model.eval()
        with torch.no_grad():
            for x, _ in batches(val, 512):
                model(x)
    s.print_summary("post-ReLU (about half the elements are exactly zero)")


def final_test(args):
    """Run ONCE, at the end of the project. Not during development."""
    train, val, test = get_data()
    _, model = run(23, seed=0, train=train, val=val, epochs=args.epochs)
    loss, acc = evaluate(model, test, 23, None, False)
    print(f"FP32 test accuracy {acc:.4f} (loss {loss:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--batch-study", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bits", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 7, 10, 23])
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="limit the sweep to these target conditions")
    p.add_argument("--batches", type=int, nargs="+", default=[128, 512, 2048],
                   help="batch sizes for --batch-study")
    p.add_argument("--updates", type=int, default=5160,
                   help="optimizer updates per run in --batch-study; 5160 is "
                        "what the 12-epoch batch-128 sweep performs")
    p.add_argument("--master", action="store_true",
                   help="use FP32 master weights in --batch-study")
    p.add_argument("--limit-train", type=int, default=None,
                   help="shrink the training set for fast iteration")
    p.add_argument("--ptq", action="store_true")
    args = p.parse_args()
    args.n_configs = len(args.bits) * 4 * 2 * args.seeds

    if args.ptq:
        ptq_check(args)
    elif args.stats:
        stats(args)
    elif args.batch_study:
        batch_study(args)
    elif args.smoke:
        smoke(args)
    elif args.test:
        final_test(args)
    else:
        sweep(args)
