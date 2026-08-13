"""Precision sweep on Tiny Shakespeare using a small CharTransformer.
Evaluates activation and weight quantization impacts on language modeling.
"""

import argparse
import csv
import math
import pathlib
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from fpbench.quantize import quantize_weights
from fpbench.activations import QuantizedActivations, ActivationStats

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
EVAL_BATCHES = 40
VAL_FRACTION = 0.1   


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


@torch.no_grad()
def evaluate(model, ids, bits, block, quant_act):
    """Calculates mean cross-entropy loss and perplexity on the validation set."""
    model.eval()
    total = 0.0
    g = torch.Generator().manual_seed(999)
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(ids, BATCH, g)
        with QuantizedActivations(model, bits if quant_act else 23, block):
            logits = model(x)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    loss = total / EVAL_BATCHES
    return loss, math.exp(loss)


def run(bits, seed, train_ids, val_ids, vocab, block=None, quant_act=False,
        quant_weight=False, steps=STEPS, log=None):
    """Executes a single training run, returning the evaluation curve and model."""
    torch.manual_seed(seed)
    model = CharTransformer(vocab).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    g = torch.Generator().manual_seed(seed)

    if quant_weight:
        quantize_weights(model, bits, block)

    curve = []
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_ids, BATCH, g)
        with QuantizedActivations(model, bits if quant_act else 23, block):
            logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if quant_weight:
            quantize_weights(model, bits, block)

        if step % EVAL_EVERY == 0 or step == steps:
            vl, ppl = evaluate(model, val_ids, bits, block, quant_act)
            curve.append({"step": step, "train_loss": loss.item(),
                          "val_loss": vl, "val_ppl": ppl})
            if log:
                print(f"  step {step:5d}  train {loss.item():.4f}  "
                      f"val {vl:.4f}  ppl {ppl:.3f}")
    return curve, model


def diagnose(args):
    """Measures and reports activation outlier statistics before sweeping."""
    train_ids, val_ids, vocab = get_data()
    print(f"vocab {vocab}, uniform-predictor perplexity {vocab}")

    print("\nTraining a short FP32 run to capture representative statistics...")
    _, model = run(23, 0, train_ids, val_ids, vocab, steps=args.steps, log=True)

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
    curve, _ = run(23, 0, train_ids, val_ids, vocab, steps=args.steps, log=True)
    dt = time.time() - t0

    best = min(curve, key=lambda r: r["val_loss"])
    print(f"\n{dt:.0f}s, {1000*dt/args.steps:.1f} ms/step")
    print(f"Final ppl {curve[-1]['val_ppl']:.3f}  "
          f"Best {best['val_ppl']:.3f} at step {best['step']}")
    print(f"\nSweep estimate: {args.n_configs} configs x {dt/60:.1f} min "
          f"= {args.n_configs * dt / 3600:.1f} h")


def sweep(args):
    """Executes the full precision configuration grid, saving metrics to CSV."""
    train_ids, val_ids, vocab = get_data()
    rows = []
    out = ROOT / "results" / "data" / "char_transformer_curves.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    conditions = [("activation", True, False), ("weight", False, True),
                  ("both", True, True)]
    formats = [("elementwise", None), ("bfp16", 16)]

    for fmt, block in formats:
        for tag, qa, qw in conditions:
            for bits in args.bits:
                for seed in range(args.seeds):
                    t0 = time.time()
                    curve, _ = run(bits, seed, train_ids, val_ids, vocab,
                                   block=block, quant_act=qa, quant_weight=qw,
                                   steps=args.steps)
                    for r in curve:
                        rows.append({"format": fmt, "block": block or 1,
                                     "target": tag, "bits": bits, "seed": seed,
                                     "vocab": vocab, **r})
                    print(f"{fmt:11s} {tag:10s} {bits:2d}b seed{seed} -> "
                          f"ppl {curve[-1]['val_ppl']:8.3f} "
                          f"({time.time()-t0:.0f}s)")
                    
                    # Rewrite continuously to prevent data loss on crash
                    with out.open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 7, 10, 23])
    args = p.parse_args()
    args.n_configs = len(args.bits) * 3 * 2 * args.seeds

    if args.diagnose:
        diagnose(args)
    elif args.smoke:
        smoke(args)
    else:
        sweep(args)