import torch, torch.nn as nn
from fpbench.quantize import round_mantissa
import csv, pathlib

rows = []

def run(bits, seed = 0, epochs = 2000):
    torch.manual_seed(seed)
    X = torch.randn(2048, 16)
    y = X @ torch.randn(16, 1)  #expected end result, @ is matrix mult
    Xq = X if bits >= 23 else round_mantissa(X, bits) #x quantized

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    #nn.Sequential chains layers together
    #nn.Linear(16, 32) - layer that learns, widens 16 numbers to 32, holds a random weight matrix and bias tensor
    #ReLU replaces negative nums with 0 - prevents two linear layers from collapsing into one
    #nn.Linear(32, 1) - collapses 32 numbers back down to one, compared against y

    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    #stochastic gradient descent, turns gradients into weight changes
    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(Xq), y)
        opt.zero_grad(); loss.backward(); opt.step() 
        #zero_grad - wipes previous gradients
        #loss.backward() - computes the gradient
        #opt.step() - applies the update to the parameter
    return loss.item()

ROOT = pathlib.Path(__file__).resolve().parents[1]

for bits in (1, 2, 3, 4, 5, 7, 10, 23):
    ratios = []
    for s in (0, 1, 2):
        base = run(23, seed=s)
        loss = run(bits=bits, seed=s)
        r = loss / base
        ratios.append(r)
        rows.append({"bits": bits, "seed": s, "loss": loss,
                     "baseline": base, "ratio": r})
    mean = sum(ratios) / len(ratios)
    print(f"{bits:2d} bits -> {mean:.4f}x baseline  "
          f"(spread {max(ratios)-min(ratios):.3f})")

out = ROOT / "results" / "data" / "train_at_vary_precision.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["bits", "seed", "loss", "baseline", "ratio"])
    w.writeheader()
    w.writerows(rows)
    