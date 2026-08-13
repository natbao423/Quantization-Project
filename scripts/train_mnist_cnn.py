"""Precision sweep on MNIST with a small CNN.

Adds three things the MLP study could not have:
  - depth, so error has somewhere to compound
  - weight sharing, so one quantized kernel's error applies at every position
  - a bounded metric (top-1 accuracy), so runs are comparable without a ratio

The whole dataset lives on the GPU. At 20k parameters the model is far too
small to hide dataloader latency, so a CPU DataLoader leaves the GPU idle most
of the time.

Usage:
    python scripts/train_mnist_cnn.py --smoke      one FP32 run, prints curves
    python scripts/train_mnist_cnn.py              the full sweep
    python scripts/train_mnist_cnn.py --test       final test accuracy, once
"""

import argparse
import csv
import pathlib
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from fpbench.quantize import round_mantissa, round_bfp, quantize_weights

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
MOMENTUM = 0.0       # see note at the bottom of this file


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
        quant_weight=False, epochs=EPOCHS, log=None):
    """One training run. Returns (per-epoch curve, trained model)."""
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # batch order is tied to the run seed, so it varies the same way weight
    # initialization does
    g = torch.Generator().manual_seed(seed)

    if quant_weight:
        quantize_weights(model, bits, block)      # round the starting weights

    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = n = 0
        for x, y in batches(train, BATCH, g, shuffle=True):
            if quant_input:
                x = quantize(x, bits, block)

            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if quant_weight:
                quantize_weights(model, bits, block)   # re-round after every step

            # accumulated on the GPU; .item() here would sync twice per step
            loss_sum += loss.detach() * y.numel()
            correct += (out.argmax(1) == y).sum()
            n += y.numel()

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
    train, val, _ = get_data(args.limit_train)
    rows = []
    out = ROOT / "results" / "data" / "mnist_cnn_curves.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    conditions = [("input", True, False), ("weight", False, True),
                  ("both", True, True)]
    formats = [("elementwise", None), ("bfp16", 16)]

    for fmt_name, block in formats:
        for tag, qi, qw in conditions:
            for bits in args.bits:
                for seed in range(args.seeds):
                    t0 = time.time()
                    curve, _ = run(bits, seed, train, val, block=block,
                                   quant_input=qi, quant_weight=qw,
                                   epochs=args.epochs)
                    for r in curve:
                        rows.append({"format": fmt_name, "block": block or 1,
                                     "target": tag, "bits": bits, "seed": seed,
                                     **r})
                    final = curve[-1]["val_acc"]
                    best = max(r["val_acc"] for r in curve)
                    print(f"{fmt_name:11s} {tag:6s} {bits:2d}b seed{seed} -> "
                          f"final {final:.4f} best {best:.4f} "
                          f"({time.time()-t0:.0f}s)")

                    with out.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)      # rewrite each run, so a crash
                                               # does not lose everything
    print(f"\nwrote {len(rows)} rows to {out}")


def final_test(args):
    """Run ONCE, at the end of the project. Not during development."""
    train, val, test = get_data()
    _, model = run(23, seed=0, train=train, val=val, epochs=args.epochs)
    loss, acc = evaluate(model, test, 23, None, False)
    print(f"FP32 test accuracy {acc:.4f} (loss {loss:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bits", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 7, 10, 23])
    p.add_argument("--limit-train", type=int, default=None,
                   help="shrink the training set for fast iteration")
    args = p.parse_args()
    args.n_configs = len(args.bits) * 3 * 2 * args.seeds

    if args.smoke:
        smoke(args)
    elif args.test:
        final_test(args)
    else:
        sweep(args)
