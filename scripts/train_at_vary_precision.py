import torch, torch.nn as nn
from fpbench.quantize import round_mantissa, quantize_weights
from fpbench.run_metadata import describe_run, save_metadata
import csv, pathlib

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

rows = []

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

#at 23 bits both quantizers are no operations, so the baseline is the same
#for all three conditions. compute it once per seed instead of 24 times.
baseline = {s: run(23, seed=s) for s in range(SEEDS)}
#changed to 10 seeds
print("FP32 baselines:", {s: round(v, 5) for s, v in baseline.items()})

def predict_zero(seed):
    torch.manual_seed(seed)
    X = torch.randn(N_SAMPLES, N_FEATURES, device=DEVICE)
    y = X @ torch.randn(N_FEATURES, 1, device=DEVICE)
    return (y ** 2).mean().item(), y.std().item()

pzero = {s: predict_zero(s)[0] for s in range(SEEDS)}

for s in range(SEEDS):
    print(f"seed {s}: predict-zero loss {pzero[s]:.2f}, "
          f"target std {predict_zero(s)[1]:.2f}")

for tag, qi, qw in [("input",  True,  False), ("weight", False, True), ("both",   True,  True)]:
    print(f"\n{tag}")
    for bits in BITS:
        ratios = []
        for s in range(SEEDS):
            base = baseline[s]
            loss = run(bits, seed = s, quant_input = qi, quant_weight = qw)
            r = loss / base
            ratios.append(r)
            rows.append({"bits": bits, "target": tag, "seed": s, "loss": loss,
             "baseline": base, "ratio": r,
             "predict_zero": pzero[s], "r2": 1 - loss / pzero[s]})
        mean = sum(ratios) / len(ratios)
        print(f"{bits:2d} bits -> {mean:.4f}x baseline  "
              f"(spread {max(ratios)-min(ratios):.3f})")

out = ROOT / "results" / "data" / "train_at_vary_precision.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["bits", "target", "seed", "loss", "baseline", "ratio", "predict_zero", "r2"])
    w.writeheader()
    w.writerows(rows)

# The CSV is written once, at the end, so the metadata follows it rather than
# bracketing the sweep the way the CNN and transformer ones do.
save_metadata(out, describe_run(
    config={"SEEDS": SEEDS, "EPOCHS": EPOCHS, "LR": LR, "BITS": list(BITS),
            "N_SAMPLES": N_SAMPLES, "N_FEATURES": N_FEATURES, "HIDDEN": HIDDEN,
            "model": "MLP 16-32-1", "optimizer": "SGD", "batching": "full-batch",
            "dataset": "synthetic y = X @ w, no noise term",
            "targets": ["input", "weight", "both"]},
    extra={"status": "complete", "rows": len(rows)}))
print(f"wrote {len(rows)} rows to {out}")