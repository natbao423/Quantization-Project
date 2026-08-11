# Quantization Project

Measuring how far each category of data in neural network training can be reduced in numerical precision before accuracy degrades.

**Headline result:** stored weights are roughly 2,150x more sensitive to low precision than input data. Inputs can be cut to a single mantissa bit and the model still explains 98.9% of target variance, degrading smoothly as a power law with no threshold. Weights re-rounded after every optimizer step hold up down to 5 bits, become unstable at 4, and fail at 3, dropping to 56% of variance explained.

## Background

Neural network training touches four categories of data: weights, activations, gradients, and optimizer state. Each could in principle be stored at a different precision. Given an exponent and `m` mantissa bits, the rounding error of a floating point number is bounded and computable, so error should be predictable per exponent. The open question is whether the tolerance for low precision is universal across models or specific to each one.

This repo covers milestone 1: a simulation that quantizes tensors to an arbitrary mantissa width and trains a model at that width. Milestone 2, reproducing the DYNASTY paper (arXiv 2210.17047, block-wise dynamic precision training), has not been started.

## Environment

- Windows, PowerShell, VS Code
- NVIDIA RTX 5070 Ti (Blackwell)
- PyTorch 2.13.0 with CUDA
- `src/` layout, editable install, package name `fpbench`

Setup:

```powershell
pip install -e .
python scripts/precision_basics.py
python scripts/train_at_vary_precision.py
pytest tests/
```

**TF32 is explicitly disabled in every script.** Blackwell GPUs run so-called FP32 matrix multiplication in TF32 by default, which carries only 10 mantissa bits instead of 23. Leaving it on would silently invalidate every FP32 baseline in this repo. Both lines below appear at the top of each script:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

## Layout

```
src/fpbench/quantize.py           quantizer
scripts/precision_basics.py       format limits and matmul error study
scripts/train_at_vary_precision.py  precision sweep during training
tests/test_quantize.py            hardware equivalence tests
results/data/*.csv                output
```

`train_at_vary_precision.csv` columns: `bits`, `target`, `seed`, `loss`, `baseline`, `ratio`, `predict_zero`, `r2`. Both reference points travel with the results, so the file is self-contained and needs nothing from the script to interpret.

The R² definition here (`1 - loss / predict_zero`) is only the standard one because targets are mean-zero by construction: `y = X @ w` with Gaussian `X` and no bias term, so the mean of `y²` and the variance of `y` agree to within sampling error. Adding a bias term or switching datasets breaks that, and R² would then need the target mean subtracted explicitly.

### `quantize.py`

- `round_mantissa(x, bits)` rounds any tensor to an arbitrary mantissa width while still storing it as FP32. It takes `floor(log2(|x|))` as the exponent, sets `scale = 2^(exp - bits)`, then computes `round(x/scale) * scale`. Zeros are restored with `torch.where` since `log2(0)` is undefined.
- `quantize_weights(model, bits)` rounds every weight matrix in place under `no_grad` using `copy_`. Biases are skipped.

### `test_quantize.py`

Verifies that `round_mantissa` is **bit-exact** against PyTorch's own `.bfloat16()` at 7 bits and `.half()` at 10 bits, using `torch.equal` on 10,000 random values.

This is the credibility claim for the whole project. Real hardware only exists at a handful of mantissa widths. Matching hardware exactly at both widths where it can be checked is what licenses trusting the simulator at 1 to 5 bits, where nothing can validate it.

## Results

### 1. PyTorch accumulates BF16 matrix multiplication in FP32

From `precision_basics.py`. Relative error against an FP64 reference, `K` being the inner dimension (the number of terms summed per output element).

| K | FP32 | BF16 in and out | BF16 rounded, FP32 math |
|---|---|---|---|
| 16 | 5.69e-08 | 2.90e-03 | 2.38e-03 |
| 64 | 1.45e-07 | 2.89e-03 | 2.37e-03 |
| 256 | 1.51e-07 | 2.88e-03 | 2.35e-03 |
| 1024 | 2.29e-07 | 3.33e-03 | 2.33e-03 |
| 4096 | 3.26e-07 | 2.86e-03 | 2.33e-03 |

BF16 error is flat across a 256x change in `K`, on both CPU and GPU. If the running sum were held in BF16, error would grow roughly as sqrt(K). It does not, so the operands are multiplied in BF16 but the sum is accumulated in FP32.

A quadrature check confirms this. The BF16 column rounds three times (both inputs plus the product), the mixed column rounds twice. Independent errors add in quadrature, so the ratio should be sqrt(3/2) = 1.225. Measured: 2.90/2.38 = 1.218.

FP32 error does grow with `K`, but far more slowly than sqrt(K), which is consistent with the library splitting long sums into partial sums rather than accumulating strictly in sequence.

**Unexplained:** GPU BF16 at K=1024 reads 3.33e-03 against roughly 2.87e-03 everywhere else. Not yet investigated.

### 2. Precision sweep during training

From `train_at_vary_precision.py`. A 2-layer MLP (Linear 16 to 32, ReLU, Linear 32 to 1; 577 parameters) trained on synthetic regression. `X = randn(2048, 16)`, `y = X @ randn(16, 1)`, with **no noise term**, so the task is exactly solvable. SGD, lr = 1e-2, 2000 full-batch epochs, 10 seeds, GPU.

Absolute loss varies several-fold across seeds, so results are reported two ways. **Ratio** is that run's loss divided by the same seed's FP32 baseline, so 1.0 means precision cost nothing. **R²** is `1 - loss / predict_zero`, where predict-zero is the loss of a model that outputs 0 for every input, so it is the fraction of target variance the model explains. R² is the more meaningful of the two and is the one to read first. All values are medians across the 10 seeds.

| mantissa bits | input ratio | input R² | weight ratio | weight R² | weight R² range |
|---|---|---|---|---|---|
| 23 (control) | 1.000 | 0.9986 | 1.00 | 0.9986 | 0.9980 to 0.9995 |
| 10 | 1.000 | 0.9986 | 1.96 | 0.9973 | 0.9962 to 0.9989 |
| 7 | 1.003 | 0.9986 | 7.54 | 0.9903 | 0.9879 to 0.9940 |
| 5 | 1.030 | 0.9986 | 7.30 | 0.9911 | 0.9861 to 0.9939 |
| 4 | 1.112 | 0.9985 | 50.30 | 0.9378 | 0.7753 to 0.9977 |
| 3 | 1.467 | 0.9980 | 301.8 | 0.5614 | 0.3864 to 0.8505 |
| 2 | 2.758 | 0.9962 | 568.8 | 0.2244 | 0.0943 to 0.3052 |
| 1 | 7.797 | 0.9894 | 632.3 | 0.0984 | 0.0474 to 0.1888 |

Reference values:

- FP32 baselines across the 10 seeds: 0.00946, 0.00955, 0.01093, 0.01233, 0.01285, 0.01366, 0.01385, 0.01746, 0.01875, 0.02263
- Predict-zero loss across the 10 seeds: 6.37, 6.45, 8.97, 10.38, 10.77, 12.69, 13.08, 16.42, 19.66, 25.28

**Why the ratio column is misleading, and R² is not.** The ratio has no fixed ceiling. Total failure to learn corresponds to predict-zero loss, which is between 495x and 2,059x baseline depending on the seed, a four-fold spread. So the same ratio means different things on different seeds, and no single ratio marks the failure line. R² is bounded above by roughly 0.9986 (the FP32 control, which is limited by 2000 SGD epochs rather than by precision) and by 0 at the point of learning nothing, on every seed. Both columns are kept because the ratio is the more sensitive measure near the top of the range, where R² is saturated and cannot resolve differences.

**Input quantization is nearly free.** Excess loss (ratio minus 1) follows 29 x 4^(-bits). The implied constant at 1 through 5 bits is 27.2, 28.1, 29.9, 28.7, 30.7. The mechanism is direct: removing one mantissa bit doubles the rounding error, and MSE squares it, so error quadruples per bit removed. There is **no elbow**, just a smooth power law all the way down to 1 bit. At 7 bits one seed scored 0.990, better than its own control, so the effect at that width is smaller than run-to-run variation.

The R² column shows this is even milder than the ratio suggests. At **1 mantissa bit**, where inputs carry two significant bits total and the loss is 7.8x baseline, the model still explains **98.9%** of target variance, against 99.86% for full FP32. A 7.8x ratio sounds like a failure and is not one. The whole input sweep, from 23 bits down to 1, moves R² by less than one percentage point.

**Weights are about 2,150x more sensitive than inputs** at 7 bits (excess loss 6.545 against 0.00303). The median is roughly flat from 10 bits down through 5, breaks sharply at 4, and fails outright at 3.

R² gives that break a scale. Weights hold above 0.99 down to 5 bits, drop to 0.94 at 4, to 0.56 at 3, and to 0.10 at 1, meaning a 1-bit-weight model explains a tenth of the variance and has essentially not learned. The elbow sits between 5 and 4 bits.

**The 5-bit versus 7-bit inversion is real.** 5-bit weights score slightly better than 7-bit on both metrics (ratio 7.30 against 7.54, R² 0.9911 against 0.9903). Since the two metrics disagree in neither direction, this is not an artifact of how loss is normalized. It is small enough to sit inside seed variation, but it does mean the weight curve is flat rather than monotone between 5 and 7 bits, and that region needs more seeds before anything is claimed about it.

**Why the asymmetry:** inputs are rounded once, before training. Weights are re-rounded after every one of the 2000 updates. Once a gradient update is smaller than half the spacing of the weight's quantization grid, the weight rounds straight back to where it started and the update vanishes. Learning stops. This is exactly why real mixed-precision training keeps a full-precision master copy of the weights and quantizes only the copy used for the forward pass.

**Quantizing both is the same as quantizing weights alone** (7.55 against 7.54 at 7 bits, median). Once the update path is broken, input precision contributes nothing.

**4-bit weights are bimodal and depend on initialization.** Per-seed ratios: 92.3, 76.8, 3.4, 26.9, 28.2, 45.3, 17.4, 111.3, 55.3, 88.5. The same runs as R²: 0.859, 0.934, 0.998, 0.980, 0.951, 0.942, 0.992, 0.775, 0.900, 0.870. Two of the ten seeds did *better* at 4 bits than at 5. This is consistent with a threshold effect governed by where the initial weights happen to land relative to the quantization grid, not with smooth degradation.

For contrast, 5-bit weights land between 0.986 and 0.994 R² on all ten seeds. One bit lower, the spread opens to 0.775 through 0.998, a range twenty times wider. Whatever 4 bits does, it does inconsistently, and a single-seed experiment at that width would report anything from near-perfect to badly broken depending on which seed was run.

## Limitations

- Only inputs and stored weights are quantized. Gradients and optimizer state are untouched.
- Storage precision only. All arithmetic is still done in FP32, and no narrow accumulator is simulated.
- Per-element exponents, not block-shared exponents. DYNASTY shares one 8-bit exponent per 4x4 block.
- Two layers, so error has almost no depth through which to compound.
- Synthetic Gaussian data, which lacks the activation-outlier structure that makes real networks hard to quantize.
- Training loss only. Nothing is held out.
- Regression with MSE, not classification accuracy. The paper reports top-1 accuracy, which saturates, and that saturation may itself be what produces the elbow shape reported there.

## Next steps

1. **Block floating point.** DYNASTY shares one 8-bit exponent per 4x4 block; the current quantizer gives every element its own exponent. Until this exists the paper's error behavior cannot be reproduced. Test it by setting block size to 1 and confirming it reduces exactly to `round_mantissa`.
2. **CIFAR-100 with ResNet-18** on the 5070 Ti (feasible; ImageNet is not). Establish an FP32 baseline and an equal-precision 8-bit block floating point baseline, the latter being the paper's own comparison point.
3. **Quantize the backward pass** with a custom `torch.autograd.Function`.
4. **DYNASTY itself:** Eq. 3b relative sensitivity, Algorithm 1 lambda tuning, EMA smoothing.
5. Loose ends: the K=1024 anomaly, and a sequential-summation comparison to show that summation order matters as much as number format.

## Open questions for the mentor

- What counts as reproducing the paper: equal-precision block floating point, or full DYNASTY?
- Are narrow accumulators in scope?
- Is CIFAR-100 alone an acceptable scope?
