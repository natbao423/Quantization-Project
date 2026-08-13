"""Precision sweep on MNIST using a small CNN.
Evaluates weight and input quantization under compounding depth and weight sharing.

Usage:
    python scripts/train_mnist_cnn.py --smoke      # Single FP32 run
    python scripts/train_mnist_cnn.py              # Full precision sweep
    python scripts/train_mnist_cnn.py --stats      # Measures activation outliers
    python scripts/train_mnist_cnn.py --test       # Final evaluation (run once)
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

# Fixed seed guarantees comparable scoring against the same validation subset
SPLIT_SEED = 12345
VAL_SIZE = 5_000

# Frozen based on FP32 baseline optimization to prevent early-stopping bias
EPOCHS = 12          
BATCH = 128
LR = 0.1

# Kept at 0.0 to prevent gradient accumulation from overriding vanishing updates
MOMENTUM = 0.0       


class SmallCNN(nn.Module):
    """Small ConvNet (~20k parameters) with two convolutional layers."""
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
    """Retrieves and normalizes MNIST data, returning GPU-resident tensors."""
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
    """Yields sequentially sliced mini-batches from GPU-resident data."""
    x, y = data
    order = (torch.randperm(len(y), generator=generator).to(DEVICE)
             if shuffle else torch.arange(len(y), device=DEVICE))
    for i in range(0, len(y), batch_size):
        j = order[i:i + batch_size]
        yield x[j], y[j]


def quantize(x, bits, block):
    """Applies elementwise or block-floating-point (BFP) quantization."""
    if bits >= 23:
        return x
    return round_mantissa(x, bits) if block is None else round_bfp(x, bits, block)


@torch.no_grad()
def evaluate(model, data, bits, block, quant_input):
    """Calculates mean cross-entropy loss and accuracy on the provided dataset."""
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
    """Executes a single training run, returning the evaluation curve and model."""
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    g = torch.Generator().manual_seed(seed)

    if quant_weight:
        quantize_weights(model, bits, block)

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
                quantize_weights(model, bits, block)

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
    """Executes a full-precision baseline to verify logic and estimate runtime."""
    train, val, _ = get_data(args.limit_train)
    n_params = sum(p.numel() for p in SmallCNN().parameters())
    print(f"Device: {DEVICE}, {n_params:,} parameters, "
          f"{len(train[1]):,} train / {len(val[1]):,} val")

    t0 = time.time()
    curve, _ = run(23, seed=0, train=train, val=val, epochs=args.epochs, log=True)
    dt = time.time() - t0

    best_acc = max(curve, key=lambda r: r["val_acc"])
    best_loss = min(curve, key=lambda r: r["val_loss"])
    print(f"\n{dt:.0f}s total, {dt/args.epochs:.1f}s/epoch")
    print(f"Final val acc {curve[-1]['val_acc']:.4f}  "
          f"Best {best_acc['val_acc']:.4f} at epoch {best_acc['epoch']}")
    print(f"Val loss bottoms at epoch {best_loss['epoch']} "
          f"({best_loss['val_loss']:.4f}), ends {curve[-1]['val_loss']:.4f}")
    print(f"Train-val accuracy gap at end: "
          f"{curve[-1]['train_acc'] - curve[-1]['val_acc']:+.4f}")
    print(f"\nFull sweep estimate: {args.n_configs} configs x {dt/60:.1f} min "
          f"= {args.n_configs * dt / 3600:.1f} h")


def sweep(args):
    """Executes the full precision configuration grid, saving metrics to CSV."""
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

                    # Rewrite continuously to prevent data loss on crash
                    with out.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)      
    print(f"\nWrote {len(rows)} rows to {out}")


def final_test(args):
    """Evaluates the model on the isolated test set (run once only)."""
    train, val, test = get_data()
    _, model = run(23, seed=0, train=train, val=val, epochs=args.epochs)
    loss, acc = evaluate(model, test, 23, None, False)
    print(f"FP32 test accuracy {acc:.4f} (loss {loss:.4f})")


def stats(args):
    """Measures pre- and post-ReLU activation outlier statistics."""
    train, val, _ = get_data()
    _, model = run(23, seed=0, train=train, val=val, epochs=args.epochs)

    # Measure pre-ReLU activations (outputs of Conv2d and Linear)
    with ActivationStats(model, block=16, types=(nn.Conv2d, nn.Linear)) as s:
        model.eval()
        with torch.no_grad():
            for x, _ in batches(val, 512):
                model(x)
    s.print_summary("MNIST CNN, pre-ReLU block exponent statistics (block=16)")

    # Measure post-ReLU activations to capture zero-handling effects
    with ActivationStats(model, block=16, types=(nn.ReLU,)) as s:
        model.eval()
        with torch.no_grad():
            for x, _ in batches(val, 512):
                model(x)
    s.print_summary("MNIST CNN, post-ReLU block exponent statistics (block=16)")


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
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()
    args.n_configs = len(args.bits) * 3 * 2 * args.seeds

    if args.stats:
        stats(args)
    elif args.smoke:
        smoke(args)
    elif args.test:
        final_test(args)
    else:
        sweep(args)