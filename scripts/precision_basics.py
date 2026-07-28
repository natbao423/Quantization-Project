import torch
import csv, pathlib

rows = []

def rel_err(ref, test):
    return ((ref - test).norm() / ref.norm()).item() #.item() turns tensor into float

print("format             eps          max          tiny")
for dt in (torch.float32, torch.bfloat16, torch.float16):
    fi = torch.finfo(dt) #returns obj with dt's numerical limits(eg. bits, precision, max)
    print(f"{str(dt):18s} {fi.eps:<12g} {fi.max:<12g} {fi.tiny:g}")

print("\nrounding behavior")
x = torch.tensor([1.0, 1.001, 1.01, 0.1, 1e-8, 1e30])
print("fp32:", x)
print("bf16:", x.to(torch.bfloat16).float())
print("fp16:", x.to(torch.float16).float())

print("\nmatrix multiplication error vs FP64")
torch.manual_seed(0)
print(f"{'K':>6} {'fp32':>10} {'bf16':>10} {'bf16 in/fp32 math':>20}")
for K in (16, 64, 256, 1024, 4096): #inner dimension, # of terms summed per output element
    A, B = torch.randn(256, K), torch.randn(K, 256) #builds two random tensors, FP32 by default
    ref = A.double() @ B.double() #cast both inputs to FP64, multiply
    A16, B16 = A.bfloat16(), B.bfloat16() #round to BF16

    e_fp32 = rel_err(ref, A @ B)                    # FP32 inputs & math
    e_bf16 = rel_err(ref, (A16 @ B16).float())      # BF16 inputs & math
    e_mixed = rel_err(ref, A16.float() @ B16.float())  # BF16 rounding, FP32 math

    print(f"{K:>6} {e_fp32:>10.2e} {e_bf16:>10.2e} {e_mixed:>20.2e}")
    rows.append({"K": K, "method": "fp32", "rel_err": e_fp32})
    rows.append({"K": K, "method": "bf16", "rel_err": e_bf16})
    rows.append({"K": K, "method": "bf16_in_fp32_math", "rel_err": e_mixed})

out = pathlib.Path("results/data/summation_order.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["K", "method", "rel_err"])
    w.writeheader()
    w.writerows(rows)