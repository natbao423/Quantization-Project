"""Precision sweep on MNIST with a small CNN.

Adds three things the MLP study could not have:
  - depth, so error has somewhere to compound
  - weight sharing, so one quantized kernel's error applies at every position
  - a bounded metric (top-1 accuracy), so runs are comparable without a ratio

The whole dataset lives on the GPU. At 20k parameters the model is far too
small to hide dataloader latency, so a CPU DataLoader leaves the GPU idle most
of the time.

Metrics
-------
Top-1 accuracy is bounded and, at 98.7% on a 5,000-image validation set, has a
standard error near 0.0016. Differences below about 0.004 are noise, which is
most of the `input` and `weight_master` table. Four unbounded metrics are
recorded alongside it. Three are measured against the FP32 model trained at the
SAME SEED, so they isolate the perturbation rather than the solution:

    kl              mean KL(FP32 || quantized) over the softmax outputs.
                    Zero at FP32, no ceiling, uses the whole distribution
                    instead of only the argmax. Locally quadratic in the logit
                    perturbation, so a 4x-per-bit error law should show up here
                    as roughly 16x per bit.
    disagree        fraction of images whose predicted class differs from the
                    FP32 model's. Strictly more sensitive than accuracy, which
                    cancels right-to-wrong flips against wrong-to-right ones.
    logit_rel_err   ||L_q - L_fp32|| / ||L_fp32||, over the whole val set.
                    Linear in the perturbation, so it should scale 4x per bit.
    logit_rel_err_c the same after removing each row's mean logit. Softmax is
                    invariant to adding a constant to every logit, so only the
                    centered figure predicts KL. The gap between the two is a
                    free readout of how much of the damage is a harmless shift.

    upd_survive     fraction of quantized weight elements whose value actually
                    changed over the epoch's optimizer steps. Not a quality
                    metric: it measures update-vanishing directly, instead of
                    inferring it from three correlated outcomes.

The FP32 reference is a separate training run, so cuDNN nondeterminism means
the 23-bit rows are NOT exactly zero. That is deliberate: those rows are the
measured noise floor for every reference-based metric, and no smaller effect in
the table should be believed.

Perplexity is not reported. For a classifier it is exp(cross_entropy) with
ten classes, so it carries no information the loss column does not, and it
compresses the entire interesting range into 1.04 to 1.5.

Modes:
    --smoke         one FP32 run, prints curves and a sweep time estimate
    --stats         activation distribution shape, for the outlier comparison
    --batch-study   does gradient noise smooth out the quantization cliff?
    --ptq           post-training quantization of an FP32 model
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
from fpbench.activations import ActivationStats, QuantizedActivations
from fpbench.provenance import build, manifest, write

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

# Where activations get rounded. "Activation" here means an intermediate
# tensor, not the activation function; ReLU itself is never modified and never
# needs to be. See --act-at.
#
#   producer  round each Conv2d output, before ReLU. Matches the transformer
#             sweep, which hooks Linear and LayerNorm outputs and so also sits
#             before its nonlinearity. Use this for cross-model comparison.
#   consumer  round each MaxPool2d output, the tensor actually handed to the
#             next weight layer. Matches what BFP hardware stores. Use this if
#             the claim is about a deployable format.
#
# For elementwise the choice is a no-op: round_mantissa preserves sign and is
# monotone, so it commutes exactly with both ReLU and MaxPool (verified, zero
# difference at 1 through 7 bits). For BFP it does not commute and roughly 17%
# of elements differ, because ReLU zeroing the block's largest-magnitude
# element changes the shared exponent. Post-ReLU blocks are about half zeros
# and their measured spread drops from 6 to 3, so `consumer` should be the
# gentler of the two. That gap is a result, not a nuisance.
ACT_HOOKS = {
    "producer": (nn.Conv2d,),
    "consumer": (nn.MaxPool2d,),
}
ACT_AT = "producer"

# The final Linear is deliberately absent from both. Its output is the logits,
# and rounding those is output quantization, not activation quantization: it
# perturbs the metric directly rather than through anything the network
# computes. The transformer sweep does hook its head, which is an
# inconsistency to note if the two models are compared on this column.


def protocol():
    """The constants a CSV cannot show, for the run manifest.

    Read at call time rather than captured at import, because --act-at
    reassigns ACT_AT after the module has loaded.
    """
    return {
        "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR, "MOMENTUM": MOMENTUM,
        "MICRO": MICRO, "SPLIT_SEED": SPLIT_SEED, "VAL_SIZE": VAL_SIZE,
        "ACT_AT": ACT_AT,
        "model": "SmallCNN", "optimizer": "SGD", "dataset": "MNIST",
        "skip_types": [t.__name__ for t in SKIP_TYPES],
        "act_hook_types": [t.__name__ for t in ACT_HOOKS[ACT_AT]],
    }


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


def quantizable_weights(model):
    """The weight tensors quantize_weights would touch, in module order.

    Factored out so QuantizedForward and the update-survival counter cannot
    drift out of step with what is actually being quantized.
    """
    for mod in model.modules():
        if isinstance(mod, SKIP_TYPES):
            continue
        w = getattr(mod, "weight", None)
        if w is not None and torch.is_floating_point(w):
            yield w


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
        for w in quantizable_weights(self.model):
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
def collect_logits(model, data, bits, block, quant_input, quant_act=False):
    """Every logit the model produces on `data`, in fixed dataset order.

    batches() without shuffle walks arange(n), so two calls on the same dataset
    line up row for row and can be differenced directly.

    Activations are rounded here as well as during training. A run that trains
    in low precision but evaluates in FP32 is measuring something else.
    """
    model.eval()
    types = ACT_HOOKS[ACT_AT]
    out = []
    for x, _ in batches(data, 512):
        if quant_input:
            x = quantize(x, bits, block)
        with QuantizedActivations(model, bits if quant_act else 23, block, types):
            out.append(model(x))
    return torch.cat(out)


@torch.no_grad()
def logit_metrics(logits, ref):
    """Compare a run's logits against the FP32 reference for the same seed.

    KL runs FP32-first, KL(P_fp32 || P_quant): it asks what the quantized model
    loses where the reference put its mass, which is the direction that
    punishes a confident wrong answer. The reverse direction would reward the
    quantized model for collapsing onto one class.
    """
    ref_logp = nn.functional.log_softmax(ref, dim=1)
    q_logp = nn.functional.log_softmax(logits, dim=1)
    # KL is nonnegative in exact arithmetic; clamped because identical
    # distributions land a few ulps below zero and a negative divergence in a
    # results table reads as a bug.
    kl = (ref_logp.exp() * (ref_logp - q_logp)).sum(1).mean().clamp_min(0)

    diff = logits - ref
    # Softmax ignores a constant added to every logit in a row, so the centered
    # figure is the part of the error that can change a prediction.
    dc = diff - diff.mean(1, keepdim=True)
    rc = ref - ref.mean(1, keepdim=True)

    return {
        "kl": kl.item(),
        "disagree": (logits.argmax(1) != ref.argmax(1)).float().mean().item(),
        "logit_rel_err": (diff.norm() / ref.norm()).item(),
        "logit_rel_err_c": (dc.norm() / rc.norm()).item(),
    }


@torch.no_grad()
def evaluate(model, data, bits, block, quant_input, ref=None, quant_act=False):
    """Return a metrics dict. `ref` is FP32 reference logits, or None.

    Returns a dict rather than a tuple because the reference-based metrics are
    optional; callers that only want loss and accuracy read two keys.
    """
    logits = collect_logits(model, data, bits, block, quant_input, quant_act)
    y = data[1]
    loss = nn.functional.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()

    out = {"val_loss": loss, "val_acc": acc}
    if ref is not None:
        out.update(logit_metrics(logits, ref))
    return out


@torch.no_grad()
def weight_snapshot(model, bits, block, master):
    """The values an optimizer step has to move to have any effect at all.

    Without a master copy the stored weights ARE the quantized values, so they
    are snapshotted directly. Rounding them again would be wrong: round_bfp
    allows carry-out and so is not strictly idempotent, and a second pass would
    report spurious changes on about 0.8% of blocks.

    With a master copy the stored weights are FP32 and always move a little, so
    the honest question is whether the value the forward pass sees moved. That
    one has to be re-derived.

    READ THE TWO COLUMNS DIFFERENTLY. Without a master, an element that does
    not move has had its update DISCARDED and will never get it back. With a
    master the update was only DEFERRED into the FP32 copy and lands once
    enough of them accumulate to cross half a grid step. Both conditions should
    show survival falling as bits drop; only the first should show accuracy
    falling with it. That divergence is the whole claim.
    """
    ws = list(quantizable_weights(model))
    if not master:
        return torch.cat([w.detach().reshape(-1).clone() for w in ws])
    return torch.cat([quantize(w.detach(), bits, block).reshape(-1) for w in ws])


def run(bits, seed, train, val, block=None, quant_input=False,
        quant_weight=False, master=False, quant_act=False, epochs=EPOCHS,
        log=None, ref=None):
    """One training run, budgeted in epochs. Returns (curve, model).

    quant_weight with master=False re-rounds the stored weights after every
    optimizer step. With master=True the stored weights stay FP32 and only the
    forward pass sees quantized values.

    quant_act rounds intermediate tensors through forward hooks, using a
    straight-through estimator. Activations cannot be rounded in place the way
    weights are: they are rebuilt every forward pass and sit inside the autograd
    graph, where rounding has zero derivative almost everywhere and would stop
    training outright. The forward value is exactly the rounded one; only the
    backward pass pretends the rounding was the identity.

    That makes `activation` comparable to `weight_master`, NOT to `weight`.
    Both compute the gradient at a quantized point and apply it to an
    unquantized quantity, so both isolate representation error. `weight` also
    discards sub-grid updates and is measuring a second thing on top.

    `ref` is the FP32 reference logits for this seed; pass None to skip the
    reference-based metrics.
    """
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # batch order is tied to the run seed, so it varies the same way weight
    # initialization does
    g = torch.Generator().manual_seed(seed)

    if quant_weight and not master:
        quantize_weights(model, bits, block)      # round the starting weights

    # Only meaningful when weights are quantized; at FP32 every element moves.
    track_updates = quant_weight and bits < 23
    act_bits = bits if quant_act else 23
    act_types = ACT_HOOKS[ACT_AT]

    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = n = 0
        moved = elems = 0
        for x, y in batches(train, BATCH, g, shuffle=True):
            if quant_input:
                x = quantize(x, bits, block)

            opt.zero_grad()
            # Both are context managers and nest cleanly: QuantizedForward
            # swaps the stored weights, QuantizedActivations attaches hooks.
            # Neither touches the other's state.
            with QuantizedActivations(model, act_bits, block, act_types):
                if quant_weight and master:
                    with QuantizedForward(model, bits, block):
                        out = model(x)
                        loss = nn.functional.cross_entropy(out, y)
                        loss.backward()
                else:
                    out = model(x)
                    loss = nn.functional.cross_entropy(out, y)
                    loss.backward()

            before = weight_snapshot(model, bits, block, master) if track_updates else None
            opt.step()
            if quant_weight and not master:
                quantize_weights(model, bits, block)   # re-round after each step
            if track_updates:
                after = weight_snapshot(model, bits, block, master)
                moved += (before != after).sum()       # stays on the GPU
                elems += before.numel()

            # accumulated on the GPU; .item() here would sync twice per step
            loss_sum += loss.detach() * y.numel()
            correct += (out.argmax(1) == y).sum()
            n += y.numel()

        if quant_weight and master:
            with QuantizedForward(model, bits, block):
                m = evaluate(model, val, bits, block, quant_input, ref, quant_act)
        else:
            m = evaluate(model, val, bits, block, quant_input, ref, quant_act)

        # An explicit schema, not **m, because the CSV writer takes its header
        # from the first row alone: any condition that produced a different
        # key order or a missing key would silently corrupt the file.
        row = {
            "epoch": epoch,
            "train_loss": loss_sum.item() / n,
            "train_acc": correct.item() / n,
            "val_loss": m["val_loss"],
            "val_acc": m["val_acc"],
            "kl": m.get("kl", ""),
            "disagree": m.get("disagree", ""),
            "logit_rel_err": m.get("logit_rel_err", ""),
            "logit_rel_err_c": m.get("logit_rel_err_c", ""),
            "upd_survive": (moved.item() / elems) if track_updates else 1.0,
        }
        curve.append(row)
        if log:
            print(f"  epoch {epoch:2d}  train {row['train_loss']:.4f}/"
                  f"{row['train_acc']:.4f}   val {m['val_loss']:.4f}/"
                  f"{m['val_acc']:.4f}"
                  + (f"   kl {m['kl']:.5f}  dis {m['disagree']:.4f}"
                     if ref is not None else "")
                  + (f"   upd {row['upd_survive']:.3f}" if track_updates else ""))
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
                        m = evaluate(model, val, bits, block, quant_input)
                else:
                    m = evaluate(model, val, bits, block, quant_input)
                curve.append({"step": step, **m})
                if log:
                    print(f"  update {step:5d}  "
                          f"val {m['val_loss']:.4f}/{m['val_acc']:.4f}")
            if step >= updates:
                break
    return curve, model

def build_references(train, val, seeds, epochs):
    """FP32 val logits per seed, for the reference-based metrics.

    One extra FP32 run per seed, roughly 12 seconds each against a sweep
    measured in hours. Pairing is strictly within seed: a run is compared only
    against the FP32 model it would have been, not against a pooled baseline.
    Comparing across seeds would fold the 0.9824-0.9860 initialization spread
    into every number.
    """
    refs = {}
    for seed in seeds:
        t0 = time.time()
        _, model = run(23, seed, train, val, epochs=epochs)
        refs[seed] = collect_logits(model, val, 23, None, False)
        m = evaluate(model, val, 23, None, False)
        print(f"reference seed{seed}: acc {m['val_acc']:.4f} "
              f"loss {m['val_loss']:.4f}  ({time.time()-t0:.0f}s)")
    return refs


def ptq_check(args):
    """Post-training quantization of an FP32 model.

    If a plain FP32 model survives 1-bit-mantissa weights, then MNIST simply
    tolerates this representation and weight_master's flat curve is real.
    If it craters, weight_master should NOT be flat, and there is a bug.
    """
    train, val, _ = get_data()
    _, model = run(23, seed=0, train=train, val=val, epochs=args.epochs)

    # The reference is this same model before any rounding, so unlike the
    # sweep there is no retraining and no cuDNN noise floor: the FP32 row is
    # exactly zero and any nonzero KL below is entirely representation error.
    ref = collect_logits(model, val, 23, None, False)
    base = evaluate(model, val, 23, None, False, ref)
    print(f"\n{'bits':>4} {'acc':>7} {'loss':>8} {'kl':>10} {'disagree':>9} "
          f"{'rel_err':>8} {'rel_err_c':>10} {'uniq':>6}")
    print(f"{'FP32':>4} {base['val_acc']:7.4f} {base['val_loss']:8.4f} "
          f"{base['kl']:10.6f} {base['disagree']:9.4f} "
          f"{base['logit_rel_err']:8.4f} {base['logit_rel_err_c']:10.4f} "
          f"{model.net[0].weight.abs().unique().numel():6d}")

    rows = []
    for bits in [10, 7, 5, 4, 3, 2, 1, 0]:
        with QuantizedForward(model, bits, None):
            nuniq = model.net[0].weight.abs().unique().numel()
            m = evaluate(model, val, bits, None, False, ref)
        m.update(bits=bits, uniq=nuniq)
        rows.append(m)
        print(f"{bits:4d} {m['val_acc']:7.4f} {m['val_loss']:8.4f} "
              f"{m['kl']:10.6f} {m['disagree']:9.4f} "
              f"{m['logit_rel_err']:8.4f} {m['logit_rel_err_c']:10.4f} "
              f"{nuniq:6d}")

    # Accuracy saturates here and cannot show a scaling law. KL and the logit
    # error can. Theory: halving the mantissa doubles the representation error,
    # so rel_err should go as 2x per bit removed and KL, being quadratic in the
    # perturbation, as 4x.
    # Normalized per bit, since the bit list is not evenly spaced: a raw 10->7
    # ratio spans three bits and is not comparable to a 5->4 ratio.
    print("\nratio per bit removed (expect ~2x for rel_err_c, ~4x for kl)")
    print(f"{'bits':>9} {'kl':>8} {'rel_err_c':>10} {'disagree':>9}")
    for a, b in zip(rows, rows[1:]):
        gap = a["bits"] - b["bits"]
        r = lambda k: (b[k] / a[k]) ** (1 / gap) if a[k] and b[k] else float("nan")
        print(f"{a['bits']:3d}->{b['bits']:<4d} {r('kl'):8.2f} "
              f"{r('logit_rel_err_c'):10.2f} {r('disagree'):9.2f}")

    out = ROOT / "results" / "data" / "mnist_cnn_ptq.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Written after the fact rather than through the context manager: this
    # check runs in seconds and writes its CSV once, so there is no partial
    # state for an entry-time manifest to describe.
    write(out, build(config=protocol(), args=args,
                     extra={"status": "complete", "rows": len(rows)}))
    print(f"\nwrote {len(rows)} rows to {out}")


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

    # (tag, quant_input, quant_weight, master, quant_act)
    conditions = [
        ("input", True, False, False, False),
        ("weight", False, True, False, False),        # no master: update-vanishing
        ("weight_master", False, True, True, False),  # FP32 master:
                                                      # representation error only
        ("both", True, True, False, False),
        ("activation", False, False, False, True),    # STE, so this is the
                                                      # like-for-like partner of
                                                      # weight_master
        # Everything at once, minus the master copy. The interesting question
        # is whether it lands on top of `weight`, which is what the three
        # earlier models all showed.
        ("act_weight", False, True, False, True),
    ]
    if args.only:
        conditions = [c for c in conditions if c[0] in args.only]
    formats = [("elementwise", None), ("bfp16", 16)]

    meta = {"conditions": [c[0] for c in conditions],
            "formats": [f[0] for f in formats],
            "n_configs": len(formats) * len(conditions) * len(args.bits) * args.seeds}

    with manifest(out, config=protocol(), args=args, extra=meta) as m:
        refs = build_references(train, val, range(args.seeds), args.epochs)

        for fmt_name, block in formats:
            for tag, qi, qw, mw, qa in conditions:
                for bits in args.bits:
                    for seed in range(args.seeds):
                        t0 = time.time()
                        curve, _ = run(bits, seed, train, val, block=block,
                                       quant_input=qi, quant_weight=qw, master=mw,
                                       quant_act=qa, epochs=args.epochs,
                                       ref=refs[seed])
                        for r in curve:
                            # act_at is recorded because producer and consumer
                            # hooks give different BFP numbers; without the column
                            # two sweeps would silently pool.
                            rows.append({"format": fmt_name, "block": block or 1,
                                         "target": tag, "act_at": ACT_AT,
                                         "bits": bits, "seed": seed, **r})
                        last = curve[-1]
                        best = max(r["val_acc"] for r in curve)
                        print(f"{fmt_name:11s} {tag:13s} {bits:2d}b seed{seed} -> "
                              f"final {last['val_acc']:.4f} best {best:.4f} "
                              f"kl {last['kl']:.5f} dis {last['disagree']:.4f} "
                              f"upd {last['upd_survive']:.3f} "
                              f"({time.time()-t0:.0f}s)")

                        with out.open("w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                            w.writeheader()
                            w.writerows(rows)      # rewrite each run, so a crash
                                                   # does not lose everything
                        # kept in step with the CSV, so an interrupted sweep
                        # still reports how far it got
                        m["rows"] = len(rows)
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

    with manifest(out, config=protocol(), args=args) as m:
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
                    m["rows"] = len(rows)
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
    m = evaluate(model, test, 23, None, False)
    print(f"FP32 test accuracy {m['val_acc']:.4f} (loss {m['val_loss']:.4f})")


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
    p.add_argument("--act-at", choices=sorted(ACT_HOOKS), default=ACT_AT,
                   help="where activations are rounded: 'producer' hooks "
                        "Conv2d outputs (pre-ReLU, matches the transformer), "
                        "'consumer' hooks MaxPool2d outputs (what BFP "
                        "hardware stores). Identical for elementwise.")
    args = p.parse_args()
    ACT_AT = args.act_at
    args.n_configs = len(args.bits) * 6 * 2 * args.seeds

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