# Quantization Project

Measuring how far each category of data in neural network training can be reduced in numerical precision before accuracy degrades.

**Headline results.** Stored weights are far more sensitive to low precision than input data, by a factor of roughly 2,150x on the MLP. Inputs survive being cut to a single mantissa bit in both models. Weights do not.

**Tolerance is not universal.** The same quantizer under the same protocol produces a sharp, initialization-dependent cliff between 5 and 4 bits on a 2-layer MLP, and smooth monotone degradation with no cliff at all on a small CNN. Failure shape is a property of the model, not of the number format.

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
pip install pytest
python -m pytest tests/
python scripts/precision_basics.py
python scripts/train_at_vary_precision.py
```

**TF32 is explicitly disabled in every script.** Blackwell GPUs run so-called FP32 matrix multiplication in TF32 by default, which carries only 10 mantissa bits instead of 23. Leaving it on would silently invalidate every FP32 baseline in this repo. Both lines below appear at the top of each script:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

## Layout

```
src/fpbench/quantize.py             quantizers: per-element and block floating point
scripts/precision_basics.py         format limits and matmul error study
scripts/train_at_vary_precision.py  precision sweep on a 2-layer MLP
scripts/train_mnist_cnn.py          precision sweep on MNIST with a small CNN
scripts/summarize_curves.py         collapses per-epoch curves into summaries
tests/test_quantize.py              bit-level properties and hardware equivalence
results/data/*.csv                  output
```

`train_at_vary_precision.csv` columns: `bits`, `target`, `seed`, `loss`, `baseline`, `ratio`, `predict_zero`, `r2`. Both reference points travel with the results, so the file is self-contained and needs nothing from the script to interpret.

The R² definition here (`1 - loss / predict_zero`) is only the standard one because targets are mean-zero by construction: `y = X @ w` with Gaussian `X` and no bias term, so the mean of `y²` and the variance of `y` agree to within sampling error. Adding a bias term or switching datasets breaks that, and R² would then need the target mean subtracted explicitly.

### `quantize.py`

- `round_mantissa(x, bits)` rounds any tensor to an arbitrary mantissa width while still storing it as FP32. Implemented by integer bit manipulation: reinterpret the float as `int32`, split off the sign bit, add a rounding bias to the magnitude, then shift the low `23 - bits` mantissa bits out and back. The sign is split off first because right-shifting a negative `int32` is an arithmetic shift and would smear the sign across the result. The bias includes the retained least significant bit, which is what makes ties round to even; without it the operation would be truncation, which biases every value toward zero and does not match bfloat16.
- `round_bfp(x, bits, block)` is block floating point: one shared exponent per `block` consecutive elements of the flattened tensor. An element whose exponent sits `g` below the block maximum keeps only `bits - g` mantissa bits and disappears once `g` exceeds `bits`. That loss is the defining behavior of the format and is why it is sensitive to outliers.
- `quantize_weights(model, bits, block=None)` rounds every weight matrix in place under `no_grad` using `copy_`. `block=None` selects per-element exponents. Biases and normalization scales are skipped.
- `block_exponent_spread(x, block)` reports `emax - emin` per block, in bits. This is the quantity that determines what BFP costs on a given tensor.

Three design decisions worth stating explicitly:

**Subnormals are flushed to signed zero.** Real BFP hardware has no subnormals. This also removes a failure in the previous float-arithmetic quantizer, whose scale factor `2^(exp - bits)` underflowed to zero below about 1e-38 and returned NaN. Weights never reach that range, so no published result was affected, but gradients will.

**`round_bfp` allows carry-out rather than saturating.** When a block's largest element rounds up across a power of two, the block exponent increments instead of the value being pinned to the top of the grid. This is what makes `block=1` reduce bit-exactly to `round_mantissa`; saturation misses on roughly 0.3% of Gaussian values. The cost is that `round_bfp` is not strictly idempotent, since a carried block re-quantizes on a coarser grid. Measured at 0.77% of blocks at 6 bits with block 16, and every unstable block is a carried block. This is a property of the format, not a bug, and the test suite asserts the precise version.

**Blocks are consecutive runs of the flattened tensor.** For a Linear weight stored `(out, in)` this groups input features within one output neuron, so the shared exponent spans terms that are summed together in the matmul. DYNASTY uses 4x4 2D tiles instead. Whether that difference matters for reproducing the paper is unresolved.

### `test_quantize.py`

Three groups of tests, 53 in total.

**Bit-level properties.** Rounding to `b` mantissa bits leaves the low `23 - b` bits of the mantissa at exactly zero, verified at 0, 1, 3, 5, 7, 10, and 17 bits. Sign is preserved, the operation is idempotent, and relative error never exceeds half a grid step. A float-arithmetic quantizer can only approach these; a bit-level one satisfies them by construction.

**Hardware equivalence.** `round_mantissa` is **bit-exact** against PyTorch's own `.bfloat16()` at 7 bits, and against `.half()` at 10 bits within float16's normal range. The float16 restriction is real and necessary: float16 also has a narrower exponent range, so below its smallest normal (6.1e-05) it goes subnormal and loses mantissa bits that `round_mantissa` does not model. On 500,000 Gaussian samples that carve-out is 18 values; on 10,000 it is usually zero, which means an unrestricted test passes by luck of sample size rather than because the equivalence is total. A companion test asserts the carve-out stays below 0.01% of samples.

This is the credibility claim for the whole project. Real hardware only exists at a handful of mantissa widths. Matching hardware exactly where it can be checked is what licenses trusting the simulator at 1 to 5 bits, where nothing can validate it. The float16 caveat narrows that claim to mantissa width, which is what the simulator actually models.

**Block floating point.** `block=1` reduces bit-exactly to `round_mantissa`. A block whose elements share one exponent is left untouched. A block containing one outlier has its small values crushed to zero, while the same values survive at `block=1`. Shape is preserved across padding for 1D, 2D, and 4D tensors.

### On the rewrite

`round_mantissa` originally computed `floor(log2(|x|))` for the exponent and then `round(x/scale) * scale`. Replacing it with bit manipulation changed no results: the sweep CSV is byte-identical before and after. The two methods agree on every normal-range value tested, across Gaussian, uniform, wide-exponent, and near-power-of-two inputs at five bit widths. The rewrite is justified by exactness, by not returning NaN on subnormals, and by providing the exponent extraction that block floating point needs, not by fixing any observed error.

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

### 2. Precision sweep on a 2-layer MLP

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

### 3. Precision sweep on MNIST with a CNN

`train_mnist_cnn.py`. A 20,490-parameter CNN (Conv 1→16, ReLU, MaxPool, Conv 16→32, ReLU, MaxPool, Linear 1568→10) on MNIST. 55,000 train / 5,000 validation, split with a fixed seed independent of the run seed so every configuration is scored against the same examples. Plain SGD, lr = 0.1, momentum 0, batch 128, 12 epochs, 3 seeds. The test set is untouched.

The epoch budget is frozen at 12 for every bit width, chosen from where FP32 validation loss bottoms out. Early stopping per run would let low-precision runs stop earlier and conflate "precision hurt the model" with "it trained for fewer epochs."

Final-epoch median validation accuracy across 3 seeds:

| bits | elementwise input | elementwise weight | BFP-16 input | BFP-16 weight |
|---|---|---|---|---|
| 23 | 0.9864 | 0.9872 | 0.9868 | 0.9868 |
| 10 | 0.9868 | 0.9872 | 0.9872 | 0.9874 |
| 7 | 0.9876 | 0.9874 | 0.9874 | 0.9862 |
| 5 | 0.9872 | 0.9860 | 0.9870 | 0.9830 |
| 4 | 0.9872 | 0.9842 | 0.9868 | 0.9760 |
| 3 | 0.9872 | 0.9780 | 0.9870 | 0.9558 |
| 2 | 0.9866 | 0.9606 | 0.9868 | 0.8826 |
| 1 | 0.9854 | 0.9254 | 0.9866 | 0.7786 |

**The cliff is gone.** This is the headline. On the MLP, weight quantization was flat from 10 bits through 5, broke sharply at 4, and failed at 3, with per-seed results at 4 bits spanning a twenty-fold range. On the CNN the same quantizer under the same protocol produces smooth monotone degradation with no elbow anywhere, and the 3-seed spread at 4 bits is 0.9836 to 0.9850. At 1 mantissa bit the CNN still classifies 92.5% of digits correctly, where the MLP had effectively stopped learning.

Two models, one quantizer, one protocol, qualitatively different failure shapes. That is direct evidence that quantization tolerance is **not universal**, which is the open question the project premise is built on. It is also, independently, a justification for the per-layer precision assignment that DYNASTY performs: if tolerance varied this much between two models, there is no reason to expect it to be uniform within one.

**Input quantization is free, more so than on the MLP.** Flat at roughly 0.987 from 23 bits down to 1, with validation loss moving only from 0.0469 to 0.0502. At 1 bit a pixel carries two significant bits and accuracy falls 0.1 points. The MLP at least showed a clean power law here; the CNN shows nothing. Pooling and weight sharing average the input rounding error away before it reaches the classifier.

**Block floating point costs what theory says it should.** BFP-16 tracks per-element exponents down to 7 bits, then separates: 0.30 accuracy points behind at 5 bits, 0.82 at 4, 2.22 at 3, 7.80 at 2, 14.68 at 1. The penalty appears exactly where mantissa bits become scarce enough that alignment shifts start destroying the smaller elements in each block. BFP never beats per-element exponents at matched width, which it cannot, since a shared exponent can only lose information; `summarize_curves.py` asserts this as a standing sanity check.

**Low precision slows convergence rather than capping it.** At 7 bits and above, weight runs peak at epoch 10 and slip about 0.15 points by epoch 12. At 4, 3, and 2 bits they peak at epoch 12 and were still improving when the budget ran out. This is a different mechanism from the MLP's update-vanishing, and it means the frozen budget is mildly unfair to low-precision runs. Best-epoch numbers are in `mnist_cnn_summary.csv`; they change no conclusion.

**Neither metric resolves anything above 7 bits.** From 23 to 7 bits every configuration sits between 0.986 and 0.988, inside seed noise, and validation loss sits at 0.046 to 0.048 for all of them including FP32. The 7-bit elementwise weight loss of 0.0455, below the FP32 baseline's 0.0467, is noise and not an effect. This is the accuracy-saturation limitation showing up in practice: with only 1.3 points of headroom above the baseline, the interesting region is 5 bits and below.


## Limitations

- Only inputs and stored weights are quantized. Gradients and optimizer state are untouched. Two of the four categories in the premise are unmeasured.
- Storage precision only. All arithmetic is still done in FP32, and no narrow accumulator is simulated.
- Both models are small. The CNN is 20k parameters and 3 layers; nothing here has the depth of a modern network.
- Neither dataset has the activation-outlier structure that makes real networks hard to quantize, and that block floating point is most sensitive to. This is the main reason to add a transformer.
- The MLP study reports training loss only, with nothing held out. The CNN study has a proper validation split; its test set is still untouched.
- Accuracy saturates. On MNIST there is only 1.3 points of headroom above the FP32 baseline, so nothing above 7 bits is resolvable by either accuracy or validation loss. The paper's own elbow may partly be this effect rather than a property of quantization.
- Only 3 seeds on the CNN, against 10 on the MLP.
- BFP blocks are consecutive runs of the flattened tensor, not DYNASTY's 4x4 tiles.

## Next steps

Block floating point, the classification metric, and the CNN are done.

1. **A small transformer on a toy corpus,** measured by perplexity. Softmax and LayerNorm produce activation outliers, which is exactly the structure both current datasets lack and exactly what block floating point is sensitive to. Log `block_exponent_spread` on activations here; it should predict where BFP hurts before the accuracy drop shows it. This is also the third model, which turns "tolerance is not universal" from a two-point observation into a trend.
2. **Quantize the backward pass** with a custom `torch.autograd.Function`. Gradients are the third of four categories in the premise and are entirely unmeasured. This is where the subnormal flushing decision in `quantize.py` stops being cosmetic.
3. **DYNASTY itself:** Eq. 3b relative sensitivity, Algorithm 1 lambda tuning, EMA smoothing. Establish the equal-precision 8-bit BFP baseline first, since that is the paper's own comparison point.
4. Loose ends: raise the CNN to 10 seeds to match the MLP, measure the momentum confound deliberately, and revisit the K=1024 anomaly and a sequential-summation comparison.

**Protocol for every sweep from here.** Fix the epoch budget from the FP32 run and reuse it at every bit width; early stopping per run conflates "precision hurt the model" with "it trained for fewer epochs." Fix the validation split independently of the run seed. Log full curves every epoch and report final and best-epoch numbers side by side. Keep the test set untouched until the end.

## Open questions for the mentor

- **Block geometry, blocking.** `round_bfp` blocks along consecutive runs of the flattened tensor; DYNASTY uses 4x4 2D tiles. For a Linear weight these roughly coincide, but for a conv weight `(out, in, kh, kw)` they do not, and a 3x3 kernel is 9 elements, which divides evenly into neither. Every CNN sweep run before this is settled would have to be discarded if it changes.
- What counts as reproducing the paper: equal-precision block floating point, or full DYNASTY?
- Are narrow accumulators in scope?
- Is MNIST plus a small transformer acceptable, or is CIFAR-100 with ResNet-18 required?
- Should normalization-layer weights be quantized? They are currently skipped, which matters once the transformer lands.
