import torch, torch.nn as nn
from fpbench.quantize import round_mantissa, quantize_weights
import csv, pathlib

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

rows = []

def run(bits, seed=0, epochs=2000, quant_input=True, quant_weight=False):
    torch.manual_seed(seed)
    X = torch.randn(2048, 16, device=DEVICE)
    y = X @ torch.randn(16, 1, device=DEVICE)  #expected end result, @ is matrix mult
    Xq = round_mantissa(X, bits) if (quant_input and bits < 23) else X

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEVICE)
    #nn.Sequential chains layers together
    #nn.Linear(16, 32) - layer that learns, widens 16 numbers to 32
    #ReLU replaces negative nums with 0 - stops two linear layers collapsing into one
    #nn.Linear(32, 1) - collapses 32 numbers back down to one, compared against y

    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

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
baseline = {s: run(23, seed=s) for s in (0, 1, 2)}
print("FP32 baselines:", {s: round(v, 5) for s, v in baseline.items()})

for s in (0, 1, 2):
    torch.manual_seed(s)
    X = torch.randn(2048, 16, device=DEVICE)
    y = X @ torch.randn(16, 1, device=DEVICE)
    print(f"seed {s}: predict-zero loss {(y ** 2).mean().item():.2f}, "
          f"target std {y.std().item():.2f}")

for tag, qi, qw in [("input",  True,  False), ("weight", False, True), ("both",   True,  True)]:
    print(f"\n{tag}")
    for bits in (1, 2, 3, 4, 5, 7, 10, 23):
        ratios = []
        for s in (0, 1, 2):
            base = baseline[s]
            loss = run(bits, seed = s, quant_input = qi, quant_weight = qw)
            r = loss / base
            ratios.append(r)
            rows.append({"bits": bits, "target": tag, "seed": s, "loss": loss, "baseline": base, "ratio": r})
        mean = sum(ratios) / len(ratios)
        print(f"{bits:2d} bits -> {mean:.4f}x baseline  "
              f"(spread {max(ratios)-min(ratios):.3f})")

out = ROOT / "results" / "data" / "train_at_vary_precision.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["bits", "target", "seed", "loss", "baseline", "ratio"])
    w.writeheader()
    w.writerows(rows)