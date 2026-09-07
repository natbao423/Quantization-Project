# Quantization Project

Measuring how far each category of data in neural network training can be reduced in numerical precision before accuracy degrades.

**What separates the sensitive case from the insensitive ones is a mechanism, not a category.** Across three models, quantizing inputs, activations, or weights-with-an-FP32-master all cost about the same. Quantizing weights *without* a master copy costs 40 to 90 times more. The difference is not which data was rounded; it is whether an update smaller than half a grid step is deferred into a full-precision copy or discarded outright.

**An earlier headline is retracted.** This README previously reported that stored weights are roughly 2,150x more sensitive than inputs, and presented that as a property of the two categories. The comparison was confounded: the weight condition re-rounds after every optimizer step and so measures representation error *plus* update-vanishing, while the input condition rounds once and measures representation error alone. Adding the `weight_master` condition, which isolates representation error in weights, closes the gap to within 2x. The 2,150x number is reproduced below and is real, but it measures the mechanism, not the category.

**Failure shape varies across models, but architecture is not isolated.** A 2-layer MLP shows a sharp, initialization-dependent cliff between 5 and 4 bits. The CNN and the transformer both degrade smoothly, with no cliff anywhere. That is a real difference, but the three models also differ in batching and optimizer (full-batch SGD, minibatch SGD, AdamW), so "quantization tolerance is architecture-specific" is **not** yet a defensible claim from this data. The MLP is the outlier, and the confound is named in Limitations.

## Background

Neural network training touches four categories of data: weights, activations, gradients, and optimizer state. Each could in principle be stored at a different precision. Given an exponent and `m` mantissa bits, the rounding error of a floating point number is bounded and computable, so error should be predictable per exponent. The open question is whether the tolerance for low precision is universal across models or specific to each one.

This repo covers milestone 1: a simulation that quantizes tensors to an arbitrary mantissa width and trains a model at that width. Weights, inputs, and activations are covered. **Gradients and optimizer state are not**, so two of the four categories remain unmeasured. Milestone 2, reproducing the DYNASTY paper (arXiv 2210.17047, block-wise dynamic precision training), has not been started.

## Environment

- Windows, PowerShell, VS Code
- NVIDIA RTX 5070 Ti (Blackwell, sm_120)
- Python 3.14, PyTorch 2.13.0 + CUDA 13.0
- `src/` layout, editable install, package name `fpbench`

Setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest tests/
```

`requirements.txt` pins third-party packages only. **`fpbench` is deliberately absent from it** and must be installed separately with `pip install -e .`. Do not regenerate the file with a bare `pip freeze`: because `fpbench` is an editable VCS install, freeze emits a `-e git+https://...@<sha>#egg=fpbench` line that clones the repo at whatever commit HEAD happened to be and installs *that* over the working tree. This actually happened, pinning the package five commits behind the quantizer rewrite. Use `pip freeze --exclude-editable`.

**TF32 is explicitly disabled in every script.** Blackwell GPUs run so-called FP32 matrix multiplication in TF32 by default, which carries only 10 mantissa bits instead of 23. Leaving it on would silently invalidate every FP32 baseline in this repo. Both lines below appear at the top of each script, and the runtime value of both flags is recorded in every run's metadata file rather than assumed:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

## Reproducing a result

Every sweep writes a `<name>.meta.json` beside its CSV recording the commit, whether the working tree was dirty, the full command line, the resolved configuration including defaults, and the machine. A CSV records only the axes a sweep varied; the metadata file records everything held fixed, which is otherwise recoverable only by reading the source at the right commit. Check `status` before trusting a file: a sweep killed partway leaves `"failed"` and a row count.

```powershell
python scripts/train_mnist_cnn.py --smoke          # one FP32 run, sanity + timing
python scripts/train_mnist_cnn.py                  # the full CNN sweep, ~50 min
python scripts/summarize_curves.py                 # collapse curves into tables
python scripts/train_char_transformer.py           # the transformer sweep, ~4 h
python scripts/train_at_vary_precision.py          # the MLP sweep
```

Note that the sweeps overwrite their canonical CSV without a guard, so a short `--steps` sanity run will destroy a completed result. Commit results before experimenting.

## Layout

```
src/fpbench/quantize.py             quantizers: per-element and block floating point
src/fpbench/activations.py          activation quantization (STE) and outlier diagnostics
src/fpbench/run_metadata.py         provenance sidecars for every results file
scripts/precision_basics.py         format limits and matmul accumulator study
scripts/train_at_vary_precision.py  precision sweep on a 2-layer MLP
scripts/train_mnist_cnn.py          precision sweep on MNIST with a small CNN
scripts/train_char_transformer.py   precision sweep on a character transformer
scripts/summarize_curves.py         collapses per-epoch curves into summaries
tests/test_quantize.py              bit-level properties and hardware equivalence
results/data/*.csv                  output, each with a .meta.json sidecar
```

`train_at_vary_precision.csv` columns: `bits`, `target`, `seed`, `loss`, `baseline`, `ratio`, `predict_zero`, `r2`. Both reference points travel with the results, so the file is self-contained and needs nothing from the script to interpret.

The R² definition here (`1 - loss / predict_zero`) is only the standard one because targets are mean-zero by construction: `y = X @ w` with Gaussian `X` and no bias term, so the mean of `y²` and the variance of `y` agree to within sampling error. Adding a bias term or switching datasets breaks that, and R² would then need the target mean subtracted explicitly.

### `quantize.py`

- `round_mantissa(x, bits)` rounds any tensor to an arbitrary mantissa width while still storing it as FP32. Implemented by integer bit manipulation: reinterpret the float as `int32`, split off the sign bit, add a rounding bias to the magnitude, then shift the low `23 - bits` mantissa bits out and back. The sign is split off first because right-shifting a negative `int32` is an arithmetic shift and would smear the sign across the result. The bias includes the retained least significant bit, which is what makes ties round to even; without it the operation would be truncation, which biases every value toward zero and does not match bfloat16.
- `round_bfp(x, bits, block)` is block floating point: one shared exponent per `block` consecutive elements of the flattened tensor. An element whose exponent sits `g` below the block maximum keeps only `bits - g` mantissa bits, and disappears once `g` reaches `bits + 2`. That loss is the defining behavior of the format and is why it is sensitive to outliers. The threshold is `bits + 2` rather than `bits` because the block's grid spacing is `2^(emax - bits)` and rounding is to nearest, so an element survives until it falls below *half* a step; values landing exactly on the half-step tie vanish one exponent earlier, since half-to-even rounds them to zero.
- `quantize_weights(model, bits, block=None)` rounds every weight matrix in place under `no_grad` using `copy_`. `block=None` selects per-element exponents. Biases and normalization scales are skipped. `bits >= 23` means "quantization off" and touches nothing — the guard lives here rather than in `round_bfp` because BFP at 23 mantissa bits is a real format that still coarsens sub-max elements, and only the sweeps overload 23 as a sentinel.
- `block_exponent_stats(x, block)` returns two per-block statistics in bits: `spread` (`emax - emin`), how many bits the smallest element loses to the alignment shift, and `headroom` (`emax - emedian`), which detects outliers. Spread alone cannot discriminate between distributions, because it is dominated by whichever element lands nearest zero — which for any continuous distribution is near zero. Headroom is the one to read for outlier structure.

Three design decisions worth stating explicitly:

**Subnormals are flushed to signed zero.** Real BFP hardware has no subnormals. This also removes a failure in the previous float-arithmetic quantizer, whose scale factor `2^(exp - bits)` underflowed to zero below about 1e-38 and returned NaN. Weights never reach that range, so no published result was affected, but gradients will.

**`round_bfp` allows carry-out rather than saturating.** When a block's largest element rounds up across a power of two, the block exponent increments instead of the value being pinned to the top of the grid. This is what makes `block=1` reduce bit-exactly to `round_mantissa`; saturation misses on roughly 0.3% of Gaussian values. The cost is that `round_bfp` is not strictly idempotent, since a carried block re-quantizes on a coarser grid. Measured at 0.77% of blocks at 6 bits with block 16, and every unstable block is a carried block. This is a property of the format, not a bug, and the test suite asserts the precise version.

**Blocks are consecutive runs of the flattened tensor.** For a Linear weight stored `(out, in)` this groups input features within one output neuron, so the shared exponent spans terms that are summed together in the matmul. DYNASTY uses 4x4 2D tiles instead. Whether that difference matters for reproducing the paper is unresolved.

The `bfp16` tag throughout means **block size 16**, not bfloat16. This is a naming mistake that has not yet been corrected in the CSVs.

### `activations.py`

Weights can be rounded in place after each optimizer step. Activations cannot: they are rebuilt on every forward pass and sit inside the autograd graph, where rounding has zero derivative almost everywhere and would stop training outright.

- `quantize_ste(x, bits, block)` uses a **straight-through estimator**. The forward value is exactly the rounded one; the backward pass pretends the rounding was the identity. Only the gradient path is a fiction.
- `QuantizedActivations` delivers this through forward hooks, so the model definition never mentions quantization.
- `ActivationStats` collects `block_exponent_stats` over a model's activations, with reference values for distributions of known outlier structure.

This makes the `activation` condition comparable to `weight_master`, **not** to `weight`. Both compute the gradient at a quantized point and apply it to an unquantized quantity, so both isolate representation error. `weight` additionally discards sub-grid updates and is measuring a second thing on top. Reading `activation` against `weight` is what produced the retracted headline.

**Hook placement matters only for BFP.** Elementwise rounding preserves sign and is monotone, so it commutes exactly with ReLU and MaxPool (verified, zero difference at 1 through 7 bits). BFP does not commute: roughly 17% of elements differ, because ReLU zeroing a block's largest element changes the shared exponent. The CNN's `--act-at producer` (default) hooks Conv2d outputs, pre-ReLU; `--act-at consumer` hooks MaxPool2d outputs, which is what BFP hardware actually stores. Only `producer` has been run, and `act_at` is a CSV column so the two cannot silently pool.

**A known inconsistency:** the CNN deliberately does not hook its final Linear, because rounding logits is output quantization rather than activation quantization. The transformer sweep, which uses the library default of `(Linear, LayerNorm)`, *does* hook its head. One of the two must change before the activation columns are compared across models.

### `test_quantize.py`

Four groups of tests, 73 in total.

**Bit-level properties.** Rounding to `b` mantissa bits leaves the low `23 - b` bits of the mantissa at exactly zero, verified at 0, 1, 3, 5, 7, 10, and 17 bits. Sign is preserved, the operation is idempotent, and relative error never exceeds half a grid step. A float-arithmetic quantizer can only approach these; a bit-level one satisfies them by construction.

**Hardware equivalence.** `round_mantissa` is **bit-exact** against PyTorch's own `.bfloat16()` at 7 bits, and against `.half()` at 10 bits within float16's normal range. The float16 restriction is real and necessary: float16 also has a narrower exponent range, so below its smallest normal (6.1e-05) it goes subnormal and loses mantissa bits that `round_mantissa` does not model. On 500,000 Gaussian samples that carve-out is 18 values; on 10,000 it is usually zero, which means an unrestricted test passes by luck of sample size rather than because the equivalence is total. A companion test asserts the carve-out stays below 0.01% of samples.

This is the credibility claim for the whole project. Real hardware only exists at a handful of mantissa widths. Matching hardware exactly where it can be checked is what licenses trusting the simulator at 1 to 5 bits, where nothing can validate it. The float16 caveat narrows that claim to mantissa width, which is what the simulator actually models.

**Block floating point.** `block=1` reduces bit-exactly to `round_mantissa`. A block whose elements share one exponent is left untouched. A block containing one outlier has its small values crushed to zero, while the same values survive at `block=1`. Shape is preserved across padding for 1D, 2D, and 4D tensors.

**Thresholds and the 23-bit sentinel.** An element `g` exponents below the block max survives at `g = bits + 1` and first vanishes at `g = bits + 2`, checked at six widths, with the exact half-step tie asserted separately because it vanishes one exponent early. `quantize_weights` is a no-op at 23 bits in both formats, and still quantizes below it. These pin down the two places the documentation was wrong: the threshold was stated a full bit too early, and `round_bfp` was silently quantizing the 23-bit rows of the BFP sweep because only `round_mantissa` no-ops there.

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
| 1 | 7.797 | 0.9893 | 632.3 | 0.0984 | 0.0474 to 0.1888 |

Reference values:

- FP32 baselines across the 10 seeds: 0.00946, 0.00955, 0.01093, 0.01233, 0.01285, 0.01366, 0.01385, 0.01746, 0.01875, 0.02263
- Predict-zero loss across the 10 seeds: 6.37, 6.45, 8.97, 10.38, 10.77, 12.69, 13.08, 16.42, 19.66, 25.28

**Why the ratio column is misleading, and R² is not.** The ratio has no fixed ceiling. Total failure to learn corresponds to predict-zero loss, which is between 495x and 2,059x baseline depending on the seed, a four-fold spread. So the same ratio means different things on different seeds, and no single ratio marks the failure line. R² is bounded above by roughly 0.9986 (the FP32 control, which is limited by 2000 SGD epochs rather than by precision) and by 0 at the point of learning nothing, on every seed. Both columns are kept because the ratio is the more sensitive measure near the top of the range, where R² is saturated and cannot resolve differences.

**Input quantization is nearly free.** Excess loss (ratio minus 1) follows 29 x 4^(-bits). The implied constant at 1 through 5 bits is 27.2, 28.1, 29.9, 28.7, 30.7. The mechanism is direct: removing one mantissa bit doubles the rounding error, and MSE squares it, so error quadruples per bit removed. There is **no elbow**, just a smooth power law all the way down to 1 bit.

At **1 mantissa bit**, where inputs carry two significant bits total and the loss is 7.8x baseline, the model still explains **98.9%** of target variance, against 99.86% for full FP32. A 7.8x ratio sounds like a failure and is not one. The whole input sweep, from 23 bits down to 1, moves R² by less than one percentage point.

**Weights are about 2,150x more sensitive than inputs** at 7 bits (excess loss 6.545 against 0.00303). The median is roughly flat from 10 bits down through 5, breaks sharply at 4, and fails outright at 3. **Read this as a mechanism result, not a category result.** The MLP has no `weight_master` condition, so representation error and update-vanishing are not separated here; the CNN sweep separates them and finds the gap is almost entirely the latter.

R² gives the break a scale. Weights hold above 0.99 down to 5 bits, drop to 0.94 at 4, to 0.56 at 3, and to 0.10 at 1, meaning a 1-bit-weight model explains a tenth of the variance and has essentially not learned. The elbow sits between 5 and 4 bits.

**The 5-bit versus 7-bit inversion is real.** 5-bit weights score slightly better than 7-bit on both metrics (ratio 7.30 against 7.54, R² 0.9911 against 0.9903). Since the two metrics disagree in neither direction, this is not an artifact of how loss is normalized. It is small enough to sit inside seed variation, but it does mean the weight curve is flat rather than monotone between 5 and 7 bits, and that region needs more seeds before anything is claimed about it.

**Why the asymmetry:** inputs are rounded once, before training. Weights are re-rounded after every one of the 2000 updates. Once a gradient update is smaller than half the spacing of the weight's quantization grid, the weight rounds straight back to where it started and the update vanishes. Learning stops. This is exactly why real mixed-precision training keeps a full-precision master copy of the weights and quantizes only the copy used for the forward pass.

**Quantizing both is the same as quantizing weights alone** (7.55 against 7.54 at 7 bits, median). Once the update path is broken, input precision contributes nothing.

**4-bit weights are bimodal and depend on initialization.** Per-seed ratios: 92.3, 76.8, 3.4, 26.9, 28.2, 45.3, 17.4, 111.3, 55.3, 88.5. The same runs as R²: 0.859, 0.934, 0.998, 0.980, 0.951, 0.942, 0.992, 0.775, 0.900, 0.870. Two of the ten seeds did *better* at 4 bits than at 5. This is consistent with a threshold effect governed by where the initial weights happen to land relative to the quantization grid, not with smooth degradation.

For contrast, 5-bit weights land between 0.986 and 0.994 R² on all ten seeds. One bit lower, the spread opens to 0.775 through 0.998, a range twenty times wider. Whatever 4 bits does, it does inconsistently, and a single-seed experiment at that width would report anything from near-perfect to badly broken depending on which seed was run.

### 3. Precision sweep on MNIST with a CNN

`train_mnist_cnn.py`. A 20,490-parameter CNN (Conv 1→16, ReLU, MaxPool, Conv 16→32, ReLU, MaxPool, Linear 1568→10) on MNIST. 55,000 train / 5,000 validation, split with a fixed seed independent of the run seed. Plain SGD, lr = 0.1, momentum 0, batch 128, 12 epochs, 3 seeds. The test set is untouched. 288 runs, about 50 minutes.

The epoch budget is frozen at 12 for every bit width, chosen from where FP32 validation loss bottoms out. Early stopping per run would let low-precision runs stop earlier and conflate "precision hurt the model" with "it trained for fewer epochs."

**Six conditions.** The design exists to separate representation error from update-vanishing:

| condition | during training | isolates |
|---|---|---|
| `input` | image rounded per batch, weights FP32 | representation error |
| `activation` | intermediate tensors rounded via STE hooks | representation error |
| `weight_master` | forward/backward quantized, FP32 master updated | representation error |
| `weight` | re-rounded in place after every step, no master | representation + update-vanishing |
| `both` | input + weight, no master | |
| `act_weight` | activation + weight, no master | |

#### Final validation accuracy, median of 3 seeds

Elementwise (per-element exponents):

| bits | input | activation | weight_master | weight | both | act_weight |
|---|---|---|---|---|---|---|
| 23 | 0.9866 | 0.9872 | 0.9870 | 0.9870 | 0.9866 | 0.9872 |
| 10 | 0.9870 | 0.9868 | 0.9874 | 0.9872 | 0.9874 | 0.9870 |
| 7 | 0.9872 | 0.9878 | 0.9868 | 0.9872 | 0.9856 | 0.9866 |
| 5 | 0.9870 | 0.9878 | 0.9870 | 0.9866 | 0.9862 | 0.9864 |
| 4 | 0.9878 | 0.9870 | 0.9864 | 0.9844 | 0.9838 | 0.9840 |
| 3 | 0.9876 | 0.9870 | 0.9866 | 0.9764 | 0.9772 | 0.9760 |
| 2 | 0.9874 | 0.9874 | 0.9872 | 0.9614 | 0.9612 | 0.9602 |
| 1 | 0.9856 | 0.9856 | 0.9872 | 0.9250 | 0.9302 | 0.9234 |

BFP, block size 16:

| bits | input | activation | weight_master | weight | both | act_weight |
|---|---|---|---|---|---|---|
| 23 | 0.9868 | 0.9870 | 0.9874 | 0.9874 | 0.9870 | 0.9878 |
| 10 | 0.9870 | 0.9872 | 0.9870 | 0.9870 | 0.9870 | 0.9866 |
| 7 | 0.9872 | 0.9868 | 0.9864 | 0.9868 | 0.9864 | 0.9862 |
| 5 | 0.9874 | 0.9870 | 0.9868 | 0.9844 | 0.9842 | 0.9842 |
| 4 | 0.9866 | 0.9874 | 0.9872 | 0.9776 | 0.9784 | 0.9772 |
| 3 | 0.9862 | 0.9876 | 0.9876 | 0.9570 | 0.9526 | 0.9540 |
| 2 | 0.9864 | 0.9870 | 0.9862 | 0.8856 | 0.8940 | 0.8826 |
| 1 | 0.9860 | 0.9860 | 0.9868 | 0.8164 | 0.7484 | 0.7314 |

**Accuracy saturates and hides most of the effect.** At 98.7% on 5,000 validation images the standard error is about 0.0016, so differences under roughly 0.004 are not measurable no matter how many seeds are run — which covers almost the entire `input`, `activation` and `weight_master` columns. Four unbounded metrics are recorded alongside accuracy, three of them against **the FP32 model trained at the same seed**, so they isolate the perturbation rather than the solution.

#### KL from the same-seed FP32 model

Mean KL(FP32 || quantized) over the softmax outputs. Zero at FP32, no ceiling.

| bits | elem input | elem activation | elem w_master | elem weight | bfp input | bfp activation | bfp w_master | bfp weight |
|---|---|---|---|---|---|---|---|---|
| 23 | 0.000088 | 0.000132 | 0.000065 | 0.000095 | 0.000117 | 0.000093 | 0.000129 | 0.000093 |
| 10 | 0.000093 | 0.000128 | 0.000102 | 0.000151 | 0.000080 | 0.000132 | 0.000210 | 0.000255 |
| 7 | 0.000212 | 0.000140 | 0.000171 | 0.001343 | 0.000109 | 0.000191 | 0.000136 | 0.002769 |
| 5 | 0.000250 | 0.000210 | 0.000277 | 0.010994 | 0.000254 | 0.000496 | 0.000397 | 0.021106 |
| 4 | 0.000332 | 0.000314 | 0.000519 | 0.026023 | 0.000488 | 0.000744 | 0.000591 | 0.046892 |
| 3 | 0.000303 | 0.000858 | 0.000933 | 0.060581 | 0.000727 | 0.002477 | 0.002407 | 0.122627 |
| 2 | 0.001695 | 0.002742 | 0.001463 | 0.135581 | 0.002149 | 0.006944 | 0.005941 | 0.392529 |
| 1 | 0.002042 | 0.005990 | 0.004579 | 0.311018 | 0.006112 | 0.027153 | 0.011569 | 0.628759 |

**The 23-bit row is a deliberate noise floor, not zero.** The FP32 reference is a separate training run, so cuDNN nondeterminism puts it at KL ≈ 1e-4 and prediction disagreement ≈ 0.001 (5 images in 5,000). Nothing smaller than that row is believable. All twelve 23-bit entries sit in a single band from 6.5e-05 to 1.3e-04, which is the check that the row means what it claims.

#### Fraction of quantized weights that move per optimizer step

`upd_survive` measures update-vanishing directly instead of inferring it. Whole-run average.

| bits | elem weight | elem w_master | bfp weight | bfp w_master |
|---|---|---|---|---|
| 10 | 0.5717 | 0.5861 | 0.4360 | 0.4607 |
| 7 | 0.3336 | 0.3588 | 0.1680 | 0.2087 |
| 5 | 0.1830 | 0.2020 | 0.0474 | 0.0848 |
| 4 | 0.1311 | 0.1384 | 0.0159 | 0.0479 |
| 3 | 0.0889 | 0.0914 | 0.0039 | 0.0268 |
| 2 | 0.0634 | 0.0566 | 0.0015 | 0.0152 |
| 1 | 0.0734 | 0.0346 | 0.0004 | 0.0097 |

Read the two kinds of column differently. Without a master, an element that does not move has had its update **discarded** and never gets it back. With a master, the update was only **deferred** into the FP32 copy and lands once enough accumulate to cross half a grid step. Both should show survival falling as bits drop; only the first should show accuracy falling with it. That divergence is the whole claim. `input` and `activation` are 1.0000 everywhere by construction, since no weights are quantized.

#### Findings

**Three of four measured conditions lie on top of each other.** `activation` and `weight_master` agree within 2x at every width, with no consistent direction (ratios span 0.61x to 2.35x across both formats). `input` sits in the same band. The fourth, `weight`, is separated by two orders of magnitude — and the separator is not a category property. Comparing `weight` against `weight_master`, which differ only in whether an FP32 master exists:

| bits | 10 | 7 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| elementwise | 1.5x | 7.9x | 39.7x | 50.1x | 64.9x | 92.7x | 67.9x |
| BFP-16 | 1.2x | 20.4x | 53.2x | 79.3x | 50.9x | 66.1x | 54.3x |

**Update-vanishing is 40 to 90 times larger than representation error** below 5 bits, stable across widths and both formats. This is what retires the earlier "weights are far more sensitive than activations" claim as a mechanism artifact.

**Losing updates is not by itself harmful.** At 10 bits only 57% of weights move per step and accuracy is untouched. Damage appears only below about 0.15 survival. It is a threshold, not a proportional cost.

**BFP at 1 bit is frozen, not degraded.** Three weights in 10,000 move per step (0.0004). That is why it lands at 0.8164 against elementwise 0.9250: 230x fewer surviving updates, not a worse representation.

**The cleanest single result.** At 1 bit elementwise, `weight_master` has *lower* update survival than `weight` (0.0346 against 0.0734) and *better* accuracy (0.9872 against 0.9250). Same visible weight motion, opposite outcomes. That is direct evidence the distinction is deferred-versus-discarded rather than how much the weights appear to move.

**Combining categories is not additive.** `act_weight` = `both` = `weight` at every width in both formats. The total is set by whichever category has update-vanishing.

**The cliff is gone, and the MLP is the exception.** Where the MLP broke sharply at 4 bits with a twenty-fold seed spread, the CNN degrades smoothly and its 3-seed spread at 4 bits is about 0.004, which is noise. The transformer below agrees. Note the confound in Limitations before reading this as an architecture effect.

**A metric trap worth knowing.** At 7 bits, elementwise `weight` has an *uncentered* relative logit error of 1.16 — logits differ from FP32 by more than their own magnitude — while accuracy is 0.9872. The centered figure is 0.0775, so about 93% of the perturbation is a per-image constant added to all ten logits, which softmax ignores. The ratio is 15-23x at 7 bits and falls to 1.0 by 1 bit, and is exactly 1.00 for `input` and `weight_master`. Cause: runs trained without a master converge to a different overall logit offset, and cross-entropy has no gradient pushing it back. **Always report `logit_rel_err_c`.**

**Low precision slows convergence rather than capping it.** At 7 bits and above, weight runs peak around epoch 10 and slip slightly by 12. At 4, 3, and 2 bits they were still improving when the budget ran out, so the frozen budget is mildly unfair to low-precision runs. Three time-to-target thresholds (90/95/98%) are reported in `mnist_cnn_summary.csv`; one threshold censors at both ends.

**Block floating point costs what theory says it should.** BFP never beats per-element exponents at matched width, which it cannot, since a shared exponent can only lose information; `summarize_curves.py` asserts this as a standing sanity check. The BFP/elementwise KL penalty on activations rises monotonically with decreasing width (1.03, 1.36, 2.36, 2.37, 2.89, 2.53, 4.53 from 10 bits to 1) — the widening signature predicted for outlier structure, and in contrast to the transformer's flat multiplier below.

### 4. Precision sweep on a character transformer

`train_char_transformer.py`. 818,048 parameters: 4 layers, 4 heads, d_model 128, context 128, tied embeddings. Tiny Shakespeare, 1M train / 111k validation characters, vocab 65, split by position rather than randomly. AdamW, lr 1e-3, batch 64, 4000 steps, 3 seeds. FP32 perplexity 5.378 to 5.495 across seeds; a uniform predictor scores 65. 144 runs, about 4 hours.

Median validation perplexity, with paired per-seed excess over that seed's own FP32 baseline in parentheses. Paired differences are used because the baselines span 0.12 perplexity.

Elementwise:

| bits | activation | weight | both |
|---|---|---|---|
| 10 | 5.387 (-0.000) | 5.420 (-0.005) | 5.418 (-0.002) |
| 7 | 5.388 (-0.000) | 5.635 (+0.246) | 5.623 (+0.235) |
| 5 | 5.373 (-0.005) | 7.268 (+1.880) | 7.382 (+1.887) |
| 4 | 5.552 (+0.122) | 9.826 (+4.332) | 10.309 (+4.920) |
| 3 | 5.897 (+0.440) | 13.276 (+7.781) | 13.772 (+8.277) |
| 2 | 7.632 (+2.194) | 18.931 (+13.543) | 19.949 (+14.571) |
| 1 | 16.419 (+10.924) | 26.456 (+20.962) | 27.712 (+22.323) |

BFP, block size 16:

| bits | activation | weight | both |
|---|---|---|---|
| 10 | 5.387 (+0.002) | 5.477 (+0.051) | 5.477 (+0.057) |
| 7 | 5.386 (-0.002) | 6.434 (+1.045) | 6.476 (+1.088) |
| 5 | 5.532 (+0.066) | 10.915 (+5.526) | 11.171 (+5.782) |
| 4 | 5.714 (+0.305) | 15.187 (+9.798) | 14.945 (+9.529) |
| 3 | 6.508 (+1.120) | 26.416 (+21.038) | 26.828 (+21.440) |
| 2 | 9.817 (+4.429) | 26.261 (+20.883) | 27.490 (+22.100) |
| 1 | 26.802 (+21.308) | 26.362 (+20.980) | 32.088 (+26.593) |

**No cliff here either.** Two of three models degrade smoothly; the MLP looks like the exception.

**Activations follow a clean 4x-per-bit law** (ratios 3.6, 5.0, 5.0 in the resolvable range). This does **not** reproduce on CNN weights, where per-bit ratios are about 1.8 unsaturated and fall toward 1.1 as centered logit error approaches its ceiling. The clean law appears to be an activation property, not a general one.

**BFP costs roughly 1.5 bits of headroom** relative to per-element exponents.

**Both = weights alone**, matching all three other models.

**Artifact: do not read the BFP weight ordering below 3 bits.** Those rows all read about 26.4, which is a collapse floor rather than a measurement.

**This sweep is behind the CNN.** It has no `weight_master` condition, so representation error and update-vanishing are *not* separated here, and it records no reference-based metrics or update survival. The CNN's finding predicts that transformer `activation` and `weight_master` should also match on a metric with no ceiling; that prediction is untested.

### 5. Outlier structure comes from the data, not the architecture

Per-block exponent statistics, block 16, measured with `--stats` and `--diagnose`. Headroom (`emax - emedian`) is the outlier detector; reference p99 values are 1.5 for uniform, 3.0 for Gaussian, 4.5 for a t-distribution with 3 degrees of freedom, 5.0 for Gaussian with 1% outliers at 10x, and 8.0 at 100x.

| module | spread med/p99 | headroom med/p99 | >8 bits |
|---|---|---|---|
| CNN conv1 | 4 / 10 | 2 / **9.0** | 6.7% |
| CNN conv2 | 6 / 13 | 3 / **7.0** | 15.1% |
| CNN linear | 6 / 12 | 2 / **4.0** | 10.3% |
| transformer, 25 of 26 modules | 4-6 / 10-12 | 1-2 / **2.0-3.0** | 3-9% |
| transformer, blocks.2.fc1 | 5 / 12 | 2 / **4.0** | 8.3% |
| *iid Gaussian reference* | 5 / 11 | 2 / *3.0* | 7.2% |

**The CNN has outlier structure and the transformer does not**, which is the opposite of the initial expectation that softmax and LayerNorm would produce outliers. The cause is the dataset: MNIST is about 80% identical background, so a block of 16 adjacent pixels crossing a stroke is background/spike/background. Tiny Shakespeare has no such structure.

**The prediction this licenses is confirmed.** With no outliers, BFP damage on the transformer is a constant multiplier (about 2.0-2.5x) rather than a widening gap; on the CNN, where outliers exist, the BFP/elementwise activation penalty widens monotonically from 1.0x at 10 bits to 4.5x at 1 bit.

Two caveats. The CNN diagnostic is a single seed-0 run and is not written to a CSV, so cuDNN nondeterminism moves the headroom p99 by about a bit between runs; the qualitative ordering is stable but the digits are not. And the original `spread` metric could not detect outliers at all — it is dominated by whichever element lands nearest zero — which is why `headroom` was added.

### 6. Production framing

**The elementwise column does not correspond to any real format.** One mantissa bit plus eight exponent bits is 10 bits per element, and no hardware stores that. Every real sub-8-bit format is block-scaled: MXFP4, MXFP8, NVFP4, INT4-with-scale. So **BFP is the production-relevant column and elementwise is a scientific control** that isolates mantissa effects from exponent-sharing effects.

**Re-indexed by actual storage cost, the ranking inverts.** Per-element cost is `bits + 2 + 8/block`. BFP at 9.5 bits/element reaches 0.9868; elementwise needs 14 bits for 0.9866, and elementwise at 10 bits/element gets only 0.9250. The tables above are indexed by *mantissa* bits, which flatters elementwise and hides this. A bits-per-element column is worth adding to `summarize_curves.py`.

**The master-weight result reproduces the design rationale for mixed-precision training.** FP32 master weights exist in production pipelines precisely because low-precision updates vanish. This is method validation, not a novel claim.

## Limitations

- **Gradients and optimizer state are never quantized.** Two of the four categories in the premise are unmeasured, so the central question is only half answered.
- **Architecture is not isolated anywhere.** The MLP is full-batch SGD, the CNN minibatch SGD, the transformer AdamW. The MLP is also the only model with a cliff. Because batching and optimizer covary with architecture, "quantization tolerance is architecture-specific" cannot be claimed from this data. Isolating it requires holding the optimizer and batching fixed across models.
- **Storage precision only.** All arithmetic is still done in FP32, and no narrow accumulator is simulated.
- **No normalization layers in the CNN**, which is the biggest risk to the outlier finding: normalization resets activation dynamic range every layer, and every production CNN has BatchNorm.
- **Accuracy saturates.** MNIST leaves only 1.3 points of headroom above the FP32 baseline, so nothing above 7 bits is resolvable by accuracy or loss. The reference-based metrics were added for this reason and should be read first.
- **Reproducibility floor is ±0.005 accuracy** from cuDNN nondeterminism, on 3 seeds. There is no `--deterministic` option yet.
- **Fixed LR with no decay.** Production LR decay shrinks late-training updates, so this setup probably *understates* update-vanishing.
- **Activation hook sets are inconsistent across models** (the transformer hooks its head; the CNN does not), so the `activation` columns are not yet comparable between them.
- The MLP study reports training loss only, with nothing held out. The CNN and transformer have proper validation splits; the CNN's test set is still untouched.
- Only 3 seeds on the CNN and transformer, against 10 on the MLP.
- BFP blocks are consecutive runs of the flattened tensor, not DYNASTY's 4x4 tiles.
- `--act-at consumer` has never been run, so the producer/consumer gap is unmeasured.

## Next steps

1. **Quantize the backward pass** with a custom `torch.autograd.Function`. Gradients are the third of four categories and are entirely unmeasured. This is where the subnormal-flushing decision in `quantize.py` stops being cosmetic, since gradients do reach the 1e-38 range that broke the old quantizer.
2. **LR sweep at fixed bit width.** The mechanism says the collapse point is a race between gradient magnitude and grid spacing, so halving the LR should shift the collapse by one bit. This converts the update-vanishing story from an explanation into a falsifiable prediction, and it is the highest-value experiment currently possible with existing code.
3. **Transformer `weight_master` and PTQ.** The CNN's central finding predicts transformer `activation` and `weight_master` should match on a ceiling-free metric. Currently untested, and it is the cheapest available replication.
4. **Resolve the head-hooking inconsistency** between the CNN and transformer, then compare activation columns across models.
5. **DYNASTY itself:** Eq. 3b relative sensitivity, Algorithm 1 lambda tuning, EMA smoothing. Establish the equal-precision 8-bit BFP baseline first, since that is the paper's own comparison point.
6. Loose ends: raise the CNN and transformer to 10 seeds, run `--act-at consumer`, run the written-but-unexecuted `--batch-study`, add a bits-per-element column, rename the `bfp16` tag to `bfp_b16`, and revisit the K=1024 accumulator anomaly.

**Protocol for every sweep from here.** Fix the epoch budget from the FP32 run and reuse it at every bit width; early stopping per run conflates "precision hurt the model" with "it trained for fewer epochs." Fix the validation split independently of the run seed. Log full curves every epoch and report final and best-epoch numbers side by side. Pair every reference-based metric within seed. Keep the test set untouched until the end.

## Open questions for the project

- **Block geometry, blocking.** `round_bfp` blocks along consecutive runs of the flattened tensor; DYNASTY uses 4x4 2D tiles. On the CNN's actual tensors this is about half clean: `linear` (10, 1568) gives blocks of 16 consecutive input features for one output neuron, but `conv1` (16, 1, 3, 3) and `conv2` (32, 16, 3, 3) have 9-element kernels, so each 16-block spans 1.78 kernels and straddles filter boundaries. Every CNN sweep run before this is settled would have to be discarded if it changes.
- What counts as reproducing the paper: equal-precision block floating point, or full DYNASTY?
- Are narrow accumulators in scope?
- Is MNIST plus a small transformer acceptable, or is CIFAR-100 with ResNet-18 required?
- Should normalization-layer weights be quantized? They are currently skipped, which matters now that the transformer has landed.
- Given that architecture is confounded with optimizer and batching, is isolating it worth the runs, or is the per-model tolerance question better answered by adding models than by controlling the confound?
