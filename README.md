# Quantization Project

Measuring how far each category of data in neural network training can be reduced in numerical precision before accuracy degrades.

**Only the precision of the stored, updated parameter matters.** This now holds on all three models. Quantizing the weights *themselves*, with no full-precision master, is by far the most expensive thing measured: 40 to 90 times the KL of every other condition on the CNN; on the MLP, 300 to 3,000 times the excess loss of the same weights kept with an FP32 master, at 1 to 4 bits; on the transformer, up to +21 perplexity where a master copy costs at most +0.6. The distinction is not which category of data was rounded but whether the rounding confines the parameter: a quantized weight is permanently pinned to a grid and any update below half a step is erased, while every other case perturbs a *computation* and leaves the parameter free to move on a later step.

**Quantizing a computation is cheap, not free.** Inputs, gradients and the forward view of the weights cost little at any width. Activations are the exception at the bottom of the range: on the transformer they are free down to about 4 bits, then cost +2.6 perplexity at 1 bit with per-element exponents and +10.3 under block floating point, still well short of what stored weights cost. Gradients are the sharpest case: block floating point annihilates up to 76% of gradient elements outright and costs nothing measurable on any of the three models, because an annihilated gradient component is a missed step rather than a lost degree of freedom.

**An earlier headline is retracted.** This README previously reported that stored weights are roughly 2,150x more sensitive than inputs, and presented that as a property of the two categories. The comparison was confounded: the weight condition re-rounds after every optimizer step and so measures representation error *plus* update-vanishing, while the input condition rounds once and measures representation error alone. Adding the `weight_master` condition, which isolates representation error in weights, closes the gap: on the CNN to within 2x of the other conditions, and on the MLP itself to *below* the cost of input quantization. The 2,150x number is reproduced below and is real, but it measures the mechanism, not the category.

**The MLP's cliff belongs to update-vanishing, not to the MLP.** Re-rounding the MLP's weights with no master produces a sharp, initialization-dependent cliff between 5 and 4 bits. With an FP32 master copy the cliff disappears: with per-element exponents R² stays at 0.994 or above on every seed down to 1 bit, and under block floating point at 0.983 or above. The CNN and the transformer degrade smoothly in every condition. The three models still differ in batching and optimizer (full-batch SGD, minibatch SGD, AdamW), so "quantization tolerance is architecture-specific" remains **not** a defensible claim from this data; but the one qualitative difference between them, the cliff, is now tied to a mechanism rather than to a model.

**A metric caveat for the transformer.** KL and disagreement against the same-seed FP32 model cannot separate damage from divergence below about KL 0.3 on the transformer, because two FP32 models that differ only in seed are already that far apart. Conditions that land under that line with unchanged perplexity (`weight_master`, `grad`) have reached a different, equally good model rather than a damaged one. Read perplexity first there; section 4 has the calibration.

## Background

Neural network training touches four categories of data: weights, activations, gradients, and optimizer state. Each could in principle be stored at a different precision. Given an exponent and `m` mantissa bits, the rounding error of a floating point number is bounded and computable, so error should be predictable per exponent. The open question is whether the tolerance for low precision is universal across models or specific to each one.

This repo covers milestone 1: a simulation that quantizes tensors to an arbitrary mantissa width and trains a model at that width. Weights, inputs, activations and gradients are covered. **Optimizer state is not**, so one of the four categories remains unmeasured — and note that the CNN and MLP both run plain SGD at momentum 0, which has no optimizer state to quantize, so that experiment belongs on the transformer. Milestone 2, reproducing the DYNASTY paper (arXiv 2210.17047, block-wise dynamic precision training), has not been started.

## Environment

- Windows, PowerShell, VS Code
- NVIDIA RTX 5070 Ti (Blackwell, sm_120); two of them since the move to a second machine
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
python scripts/train_char_transformer.py           # the transformer sweep, ~1.7 h on two GPUs
python scripts/train_at_vary_precision.py          # the MLP sweep, ~20 min on two GPUs
python scripts/plot_results.py --format png pdf    # every figure, CSV twins, captions.txt
```

Figures read only committed results files and write a CSV twin of exactly what each one plots, plus `captions.txt` with the caption and source for every figure. PNG and PDF are both committed; PDFs carry no creation date, so an unchanged figure re-renders to an identical file.

Sweeps refuse to overwrite a results file that looks finished — one whose metadata says `complete`, or one with no metadata at all, since every CSV committed before run metadata existed has no sidecar. Write elsewhere with `--out PATH`, or pass `--force`. A file whose metadata says `failed` or `running` is a known-partial result and is overwritten with a note, because rerunning it is the point.

### Multiple GPUs

Every sweep takes `--devices`, default `auto`: each discrete GPU gets one worker process, and each worker picks up the next run as soon as it finishes the last. With **one GPU** there are no workers: everything runs in the launching process, one run at a time, as the sweeps always did. With **no CUDA GPU** it falls back to the CPU and says so. Integrated GPUs are never used by `auto`. An Intel or AMD iGPU is invisible to CUDA anyway; an NVIDIA integrated part reports itself as such and is skipped.

```powershell
python scripts/train_mnist_cnn.py                  # every discrete GPU
python scripts/train_mnist_cnn.py --devices 1      # only GPU 1 (GPU 0 drives the display here)
python scripts/train_mnist_cnn.py --jobs-per-gpu 2 # two concurrent runs per GPU
```

Rows are written in grid order, not completion order, so the CSV has the same layout however the runs were spread. **The numbers do not depend on it either:** checked by running each sweep on the pre-change code, then on one GPU, then on two. The MLP gives byte-identical CSVs. So does the CNN with cuDNN made deterministic. Left nondeterministic, the CNN differs by exactly as much as two runs of the same code differ. The list of devices actually used is recorded as `device_list` in each run's metadata. Only the main sweeps are spread; `--smoke`, `--stats`, `--ptq`, `--batch-study` and `--diagnose` run in one process on the first device.

`--dry-run` prints the plan, the run count and a measured runtime estimate without training anything. Every sweep also writes its progress into the metadata sidecar after each cell, so a long run can be watched from another terminal:

```powershell
python -c "import json;d=json.load(open('results/data/mnist_cnn_curves.meta.json'));print(d['status'],d.get('progress'))"
```

## Layout

```
src/fpbench/quantize.py             quantizers: per-element and block floating point
src/fpbench/activations.py          activation quantization (STE) and outlier diagnostics
src/fpbench/run_metadata.py         provenance sidecars for every results file
src/fpbench/cli.py                  shared sweep flags, overwrite guard, progress/ETA
src/fpbench/parallel.py             device selection and the per-GPU worker pool
src/fpbench/metrics.py              KL, disagreement and logit error against an FP32 reference
scripts/precision_basics.py         format limits and matmul accumulator study
scripts/train_at_vary_precision.py  precision sweep on a 2-layer MLP
scripts/train_mnist_cnn.py          precision sweep on MNIST with a small CNN
scripts/train_char_transformer.py   precision sweep on a character transformer
scripts/summarize_curves.py         collapses per-epoch curves into summaries
scripts/plot_results.py             result figures, CSV twins and captions
tests/test_quantize.py              bit-level properties and hardware equivalence
tests/test_parallel.py              device fallback rules and the worker pool
tests/test_activations.py           activation hooks, including the output-layer exclusion
tests/test_metrics.py               reference metrics and the transformer's validation tiling
tests/test_mlp.py                   the MLP's weight_master condition and update survival
tests/test_cli.py, test_time_to_target.py   sweep flags, overwrite guard, time-to-target
results/data/*.csv                  output, each with a .meta.json sidecar
```

`train_at_vary_precision.csv` columns: `format`, `block`, `bits`, `target`, `seed`, `loss`, `baseline`, `ratio`, `predict_zero`, `r2`, `grad_survive`, `upd_survive`. Both reference points travel with the results, so the file is self-contained and needs nothing from the script to interpret. Files written before the format column existed hold per-element rows only; readers treat a missing `format` as `elementwise`.

The R² definition here (`1 - loss / predict_zero`) is only the standard one because targets are mean-zero by construction: `y = X @ w` with Gaussian `X` and no bias term, so the mean of `y²` and the variance of `y` agree to within sampling error. Adding a bias term or switching datasets breaks that, and R² would then need the target mean subtracted explicitly.

### `quantize.py`

- `round_mantissa(x, bits)` rounds any tensor to an arbitrary mantissa width while still storing it as FP32. Implemented by integer bit manipulation: reinterpret the float as `int32`, split off the sign bit, add a rounding bias to the magnitude, then shift the low `23 - bits` mantissa bits out and back. The sign is split off first because right-shifting a negative `int32` is an arithmetic shift and would smear the sign across the result. The bias includes the retained least significant bit, which is what makes ties round to even; without it the operation would be truncation, which biases every value toward zero and does not match bfloat16.
- `round_bfp(x, bits, block)` is block floating point: one shared exponent per `block` consecutive elements of the flattened tensor. An element whose exponent sits `g` below the block maximum keeps only `bits - g` mantissa bits, and disappears once `g` reaches `bits + 2`. That loss is the defining behavior of the format and is why it is sensitive to outliers. The threshold is `bits + 2` rather than `bits` because the block's grid spacing is `2^(emax - bits)` and rounding is to nearest, so an element survives until it falls below *half* a step; values landing exactly on the half-step tie vanish one exponent earlier, since half-to-even rounds them to zero.
- `quantize_weights(model, bits, block=None)` rounds every weight matrix in place under `no_grad` using `copy_`. `block=None` selects per-element exponents. Biases and normalization scales are skipped. `bits >= 23` means "quantization off" and touches nothing — the guard lives here rather than in `round_bfp` because BFP at 23 mantissa bits is a real format that still coarsens sub-max elements, and only the sweeps overload 23 as a sentinel.
- Both quantizers take `stochastic=True`, which replaces round-to-nearest with an unbiased random dither: an element sitting a fraction `f` of a step above the grid rounds up with probability exactly `f`, so `E[round(x)] == x`. Measured at **16x less per-element bias** than round-to-nearest, at every width and in both formats. The point is not accuracy but accumulation: round-to-nearest is deterministic, so a value systematically below half a step is discarded on every single step, which is the same update-vanishing that separates `weight` from `weight_master`. Stochastic rounding turns that discard into a deferral. It costs variance -- a *single* stochastic draw is further from the original than round-to-nearest is -- so the benefit appears only across many steps, and any one-shot metric will rank it worse. Seeded from the global RNG by default, which the sweeps already set per run; an explicit generator must sit on the same device as the tensor.
- `quantize_grads(model, bits, block=None, stochastic=False)` rounds weight gradients in place, between `backward()` and `step()`. Same tensors as `quantize_weights`, so the `grad` and `weight` columns are comparable rather than covering different parameters. Unlike weights there is no accumulator a discarded contribution can be recovered from, which is why the rounding rule was expected to matter and why `stochastic=` exists; section 7 reports that it does not.
- `quantize(x, bits, block=None)` is the one entry point for "maybe quantize": it returns `x` untouched at 23 bits and otherwise dispatches to the two formats. The scripts used to carry three private copies of it.
- `quantizable_weights(model)` yields each tensor **once**. The transformer ties its output head to its token embedding, and walking modules alone yielded that tensor twice, so it was rounded twice per step and its gradient quantized twice. Elementwise rounding is idempotent, so that changed nothing there. BFP is not, because of carry-out. Rerun after the fix, the BFP transformer's `weight` perplexity moved by about 0.2% at 4 bits. Transformer results up to commit 6059c87 carry that small error; the current ones do not.
- `QuantizedForward` (the FP32-master arrangement behind `weight_master`), `weight_snapshot` (the before-and-after view behind `upd_survive`) and `GradSurvival` (`quantize_grads` plus the `grad_survive`/`grad_cos` bookkeeping) live here rather than in the scripts, so all three models use the same code for the same condition.
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

**Output layers are never rounded.** Rounding logits is output quantization rather than activation quantization: it perturbs the metric directly instead of anything the network computes from. The CNN never hooked its final Linear; the transformer did until commit 7c98bb9, and now passes its head to `QuantizedActivations(exclude=...)`. That exclusion halved the transformer's 1-bit activation damage (section 4), so the two are now comparable in what they leave out. They are still not identical in what they round: the transformer also rounds LayerNorm outputs, and the CNN has no LayerNorm.

### `test_quantize.py`

The groups below cover the bit-level core. Later groups cover stochastic rounding, gradient quantization, and the machinery shared with the scripts.

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

All MLP results are in `train_at_vary_precision.csv`: six conditions (`input`, `weight`, `weight_master`, `both`, `grad`, `grad_sr`), both formats, 8 widths, 10 seeds, 960 runs from commit 7c98bb9. That sweep reproduces every row of the earlier MLP files exactly, to the last digit of the loss, so the numbers quoted from them below stand. The tables in this section are per-element.

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

**Weights are about 2,150x more sensitive than inputs** at 7 bits (excess loss 6.545 against 0.00303). The median is roughly flat from 10 bits down through 5, breaks sharply at 4, and fails outright at 3. **Read this as a mechanism result, not a category result.** The `weight_master` condition below separates the two mechanisms on this model, and the gap is almost entirely update-vanishing.

R² gives the break a scale. Weights hold above 0.99 down to 5 bits, drop to 0.94 at 4, to 0.56 at 3, and to 0.10 at 1, meaning a 1-bit-weight model explains a tenth of the variance and has essentially not learned. The elbow sits between 5 and 4 bits.

**The 5-bit versus 7-bit inversion is real.** 5-bit weights score slightly better than 7-bit on both metrics (ratio 7.30 against 7.54, R² 0.9911 against 0.9903). Since the two metrics disagree in neither direction, this is not an artifact of how loss is normalized. It is small enough to sit inside seed variation, but it does mean the weight curve is flat rather than monotone between 5 and 7 bits, and that region needs more seeds before anything is claimed about it.

**Why the asymmetry:** inputs are rounded once, before training. Weights are re-rounded after every one of the 2000 updates. Once a gradient update is smaller than half the spacing of the weight's quantization grid, the weight rounds straight back to where it started and the update vanishes. Learning stops. This is exactly why real mixed-precision training keeps a full-precision master copy of the weights and quantizes only the copy used for the forward pass.

**Quantizing both is the same as quantizing weights alone** (7.55 against 7.54 at 7 bits, median). Once the update path is broken, input precision contributes nothing.

**4-bit weights are bimodal and depend on initialization.** Per-seed ratios: 92.3, 76.8, 3.4, 26.9, 28.2, 45.3, 17.4, 111.3, 55.3, 88.5. The same runs as R²: 0.859, 0.934, 0.998, 0.980, 0.951, 0.942, 0.992, 0.775, 0.900, 0.870. Two of the ten seeds did *better* at 4 bits than at 5. This is consistent with a threshold effect governed by where the initial weights happen to land relative to the quantization grid, not with smooth degradation.

For contrast, 5-bit weights land between 0.986 and 0.994 R² on all ten seeds. One bit lower, the spread opens to 0.775 through 0.998, a range twenty times wider. Whatever 4 bits does, it does inconsistently, and a single-seed experiment at that width would report anything from near-perfect to badly broken depending on which seed was run.

#### With an FP32 master copy

`weight_master` rounds only the copy of the weights the forward and backward pass see; the optimizer updates an FP32 master. `upd_survive` is the fraction of quantized weight elements whose value moves per step, over the whole run. Medians of 10 seeds, per-element exponents:

| bits | weight R² | weight_master R² | weight_master R² range | weight_master ratio | upd_survive weight | upd_survive weight_master |
|---|---|---|---|---|---|---|
| 10 | 0.9973 | 0.9986 | 0.9980 to 0.9995 | 1.000 | 0.2196 | 0.2696 |
| 7 | 0.9903 | 0.9986 | 0.9980 to 0.9995 | 0.999 | 0.0408 | 0.0776 |
| 5 | 0.9911 | 0.9986 | 0.9980 to 0.9994 | 1.002 | 0.0130 | 0.0379 |
| 4 | 0.9378 | 0.9985 | 0.9980 to 0.9994 | 1.021 | 0.0049 | 0.0321 |
| 3 | 0.5614 | 0.9984 | 0.9975 to 0.9992 | 1.095 | 0.0019 | 0.0306 |
| 2 | 0.2244 | 0.9978 | 0.9952 to 0.9991 | 1.652 | 0.0005 | 0.0321 |
| 1 | 0.0984 | 0.9962 | 0.9942 to 0.9975 | 3.117 | 0.0001 | 0.0345 |

**The prediction held: no cliff.** The script recorded it before the run: if the 4-bit cliff is update-vanishing, `weight_master` should not have it. At 4 bits the ten seeds span 0.9980 to 0.9994 against 0.775 to 0.998 without a master, and with per-element exponents R² stays at 0.994 or above on every seed down to 1 bit. Under BFP the same holds, at 0.9919 median and 0.9826 worst seed at 1 bit, against 0.163 for `weight`.

**Update-vanishing is 300 to 3,000 times larger than representation error here**, at 1 to 4 bits, in excess loss (ratio minus 1): 49.3 against 0.021 at 4 bits, 300.8 against 0.095 at 3, 631.3 against 2.12 at 1. At 5 and 7 bits `weight_master` is indistinguishable from FP32.

**A master copy is cheaper than quantizing the inputs.** `weight_master` costs less than `input` at every width: excess loss 0.021 against 0.112 at 4 bits, 2.12 against 6.80 at 1. On the CNN the two sit within 2x of each other in KL; here the master-weight forward pass is consistently the cheaper of the two.

**Survival tracks the outcome on this model, unlike the CNN.** With a master, 3 to 4% of weights move per step at every width from 5 bits down; without one, survival falls to 0.5% at 4 bits and 0.01% at 1. The CNN's sharper result, a master that moves *fewer* weights and still trains better, does not recur here: on the MLP the master always moves more.

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

`train_char_transformer.py`. 818,048 parameters: 4 layers, 4 heads, d_model 128, context 128, tied embeddings. Tiny Shakespeare, 1M train / 111k validation characters, vocab 65, split by position rather than randomly. AdamW, lr 1e-3, batch 64, 4000 steps, 3 seeds. FP32 perplexity 5.386 to 5.514 across seeds; a uniform predictor scores 65.

Results are from `char_transformer_curves.csv`, commit 7c98bb9: six conditions (`activation`, `weight`, `act_weight`, `weight_master`, `grad`, `grad_sr`), both formats, 8 widths, 3 seeds, 288 runs in 1.7 h on two GPUs. Three things changed from the sweep this replaces. It has a `weight_master` and the two gradient conditions. Activation rounding excludes the output layer, so logits are never rounded. Evaluation covers the whole validation split, tiled into 871 fixed windows, so every run scores exactly the same characters and the reference metrics pair one to one.

Median validation perplexity, with the paired per-seed excess over that seed's own FP32 run in parentheses. Paired differences are used because the FP32 runs span 0.13 perplexity.

Elementwise:

| bits | activation | weight | weight_master | act_weight |
|---|---|---|---|---|
| 10 | 5.394 (+0.000) | 5.409 (-0.011) | 5.391 (-0.002) | 5.407 (-0.012) |
| 7 | 5.394 (-0.002) | 5.654 (+0.234) | 5.417 (+0.021) | 5.656 (+0.261) |
| 5 | 5.375 (-0.016) | 7.278 (+1.881) | 5.444 (-0.061) | 7.378 (+1.864) |
| 4 | 5.493 (+0.025) | 9.876 (+4.362) | 5.374 (-0.040) | 9.853 (+4.339) |
| 3 | 5.663 (+0.264) | 13.347 (+7.833) | 5.403 (-0.004) | 13.635 (+8.121) |
| 2 | 6.136 (+0.746) | 18.984 (+13.589) | 5.442 (+0.047) | 19.451 (+14.064) |
| 1 | 7.978 (+2.591) | 26.498 (+20.984) | 5.617 (+0.138) | 26.998 (+21.500) |

BFP, block size 16:

| bits | activation | weight | weight_master | act_weight |
|---|---|---|---|---|
| 10 | 5.393 (+0.002) | 5.478 (+0.035) | 5.388 (+0.001) | 5.482 (+0.029) |
| 7 | 5.382 (-0.013) | 6.623 (+1.228) | 5.408 (-0.005) | 6.480 (+1.084) |
| 5 | 5.502 (+0.027) | 11.185 (+5.799) | 5.433 (+0.016) | 11.172 (+5.777) |
| 4 | 5.655 (+0.238) | 14.855 (+9.459) | 5.402 (+0.004) | 15.573 (+10.178) |
| 3 | 6.252 (+0.774) | 26.854 (+21.340) | 5.457 (+0.061) | 26.885 (+21.371) |
| 2 | 8.519 (+3.005) | 26.298 (+20.911) | 5.606 (+0.191) | 27.136 (+21.750) |
| 1 | 15.818 (+10.304) | 26.179 (+20.774) | 5.973 (+0.578) | 28.354 (+22.968) |

Per-seed perplexity scatters by up to about ±0.1 between runs that reach equally good models: `weight_master` at 5 to 10 bits spans -0.107 to +0.049. Excesses inside that band are not results.

#### Fraction of quantized weights that move per optimizer step

Medians of 3 seeds, over the last 250-step window of the run.

| bits | elem weight | elem w_master | bfp weight | bfp w_master |
|---|---|---|---|---|
| 10 | 0.8732 | 0.8787 | 0.7105 | 0.7260 |
| 7 | 0.3983 | 0.4693 | 0.0604 | 0.1710 |
| 5 | 0.0962 | 0.1997 | 0.0013 | 0.0437 |
| 4 | 0.0262 | 0.1204 | 0.0001 | 0.0223 |
| 3 | 0.0070 | 0.0707 | 0.0000 | 0.0118 |
| 2 | 0.0008 | 0.0404 | 0.0000 | 0.0068 |
| 1 | 0.0000 | 0.0227 | 0.0000 | 0.0043 |

#### Findings

**A master copy keeps quantized weights near FP32 at every width.** `weight_master` stays within +0.14 perplexity of FP32 down to 1 bit with per-element exponents and within +0.58 under BFP, while `weight` reaches +21. The CNN's central result reproduces on a second model, with a different optimizer.

**Without a master, BFP weights freeze.** At 3 bits and below not one weight element in 10,000 moves per step, and perplexity sits at a collapse floor of about 26.2 to 26.9 whatever the width. With a master the same rounding still moves 0.4 to 1.2% of weights per step and the model trains normally. The same deferred-versus-discarded split as the CNN.

**Calibrating the reference metrics: they saturate at the distance between two seeds.** Three FP32 runs that differ only in seed were compared pairwise on the same validation characters: KL 0.29 to 0.33, disagreement 27 to 28%, centered logit error 0.59 to 0.62. On this model and budget, that is how far apart two equally good models are. (Measured outside the sweep, with the sweep's own `run`, `collect_logits` and `logit_metrics`; it reproduces the sweep's FP32 perplexities exactly, but is not yet a committed results file.) Measured against it:

- `weight_master` reaches KL 0.11 to 0.28 at 4 bits and below, and `grad` at most 0.23. Both are under the cross-seed line, with perplexity unchanged. These runs have diverged to a different, equally good model, not degraded. Read perplexity for them, not KL.
- `weight` and `act_weight` reach KL 0.82 to 1.9 at 4 bits and below, about 2.5 to 6 times the cross-seed line. That is real damage, and perplexity agrees.
- Activations cross the line at 2 bits per-element (KL 0.33) and at 3 bits under BFP (0.32), and reach 0.60 and 1.31 at 1 bit.

The consequence: on this model, ratios of KL against `weight_master` say little below about 5 bits, because `weight_master`'s KL is capped by divergence rather than set by damage. Whether the CNN has the same ceiling has not been measured: its 23-bit noise floor (KL near 1e-4) compares runs of the *same* seed, not different ones. Its `weight_master` KL never exceeds 0.012, far below the transformer's line, but how far apart two CNN seeds land is not known. On this model, read perplexity first.

**Activations hold to about 4 bits, then cost more than weights with a master.** Per-element activations are within noise down to 4 bits (+0.025) and cost +0.26, +0.75 and +2.59 perplexity at 3, 2 and 1 bit. Under BFP the same costs start a bit earlier: +0.24 at 4 bits, +10.3 at 1. Below 4 bits activations cost more than `weight_master` at every width, in both formats. The CNN's finding that the two agree within 2x does not carry over at the bottom of the range: on the transformer, activations are the more sensitive of the computation-side categories.

**Excluding the logits mattered.** In the replaced sweep, which rounded the output layer's logits along with every other Linear output, 1-bit activations scored 16.0 perplexity per-element and 25.4 under BFP; with the logits unrounded they score 7.98 and 15.8. About half the apparent 1-bit activation damage was rounding the metric's own input. `weight`, `weight_master` and the gradient conditions do not round activations, and reproduce their earlier values to within run-to-run jitter.

**Activation cost grows about 3 to 4x per bit removed**, in excess perplexity: 2.8 and 3.5x per-element from 3 bits down, 3.3, 3.9 and 3.4x under BFP from 4 bits down. The earlier sweep reported a cleaner 4x-to-5x law; that included the logit rounding.

**No cliff.** Every condition degrades smoothly, and the three seeds agree closely at every width: `weight` at 4 bits spans 9.54 to 10.12 per-element, against the MLP's 0.775 to 0.998 R² there.

**BFP costs roughly 1 to 1.5 bits** relative to per-element exponents: BFP activations at 3 bits cost about what per-element ones do at 2, and BFP weights at 5 bits land between per-element 4 and 3.

**Activation + weight = weights alone**, matching the other models. This condition was tagged `both` until it was renamed to `act_weight`, the CNN's name for the same thing; on the CNN, `both` means input + weight. Files from before the rename still say `both`.

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

**The prediction this licenses is confirmed.** With no outliers, the BFP penalty on transformer activations stays between 1.7x and 3.2x of per-element KL at every width, with no widening trend; on the CNN, where outliers exist, the BFP/elementwise activation penalty widens monotonically from 1.0x at 10 bits to 4.5x at 1 bit. (The transformer range was re-measured with the logits unrounded; the earlier figure of 2.0 to 2.5x included them. The `--diagnose` statistics in the table above still include the output layer.)

Two caveats. The CNN diagnostic is a single seed-0 run and is not written to a CSV, so cuDNN nondeterminism moves the headroom p99 by about a bit between runs; the qualitative ordering is stable but the digits are not. And the original `spread` metric could not detect outliers at all — it is dominated by whichever element lands nearest zero — which is why `headroom` was added.

### 6. Production framing

**The elementwise column does not correspond to any real format.** One mantissa bit plus eight exponent bits is 10 bits per element, and no hardware stores that. Every real sub-8-bit format is block-scaled: MXFP4, MXFP8, NVFP4, INT4-with-scale. So **BFP is the production-relevant column and elementwise is a scientific control** that isolates mantissa effects from exponent-sharing effects.

**Re-indexed by actual storage cost, the ranking inverts.** Per-element cost is `bits + 2 + 8/block`. BFP at 9.5 bits/element reaches 0.9868; elementwise needs 14 bits for 0.9866, and elementwise at 10 bits/element gets only 0.9250. The tables above are indexed by *mantissa* bits, which flatters elementwise and hides this. A bits-per-element column is worth adding to `summarize_curves.py`.

**The master-weight result reproduces the design rationale for mixed-precision training.** FP32 master weights exist in production pipelines precisely because low-precision updates vanish. This is method validation, not a novel claim.

### 7. Gradient quantization is free, on all three models

`train_mnist_cnn.py --only grad grad_sr` (96 runs); the `grad` and `grad_sr`
rows of `train_at_vary_precision.csv`, which reproduce the earlier
`train_at_vary_precision_weight_grad_grad_sr.csv` exactly; and the same rows of
`char_transformer_curves.csv`.

Gradients are quantized in place between `backward()` and `step()`, over the
same tensors `quantize_weights` touches so the columns are comparable. Two
conditions: `grad` rounds to nearest, `grad_sr` uses an unbiased stochastic
dither. The pair was designed as the backward-path mirror of
`weight`/`weight_master` — if update-vanishing were the general mechanism, an
unbiased dither should close a large gap the way an FP32 master does.

**CNN, median of 3 seeds.** Accuracy across all 32 configurations spans
0.9864–0.9882, entirely inside the ±0.004 noise band:

| bits | elem acc | elem KL | bfp acc | bfp KL | bfp grad_survive |
|---|---|---|---|---|---|
| 23 | 0.9870 | 0.000092 | 0.9876 | 0.000114 | 1.0000 |
| 7 | 0.9876 | 0.000144 | 0.9870 | 0.000124 | 0.8083 |
| 4 | 0.9866 | 0.000140 | 0.9868 | 0.000126 | 0.6763 |
| 2 | 0.9872 | 0.000228 | 0.9864 | 0.000611 | 0.5166 |
| 1 | 0.9878 | 0.000546 | 0.9882 | 0.001718 | **0.4010** |

**MLP, median of 10 seeds**, FP32 control R² = 0.9986, with the `weight` column
alongside for scale:

| bits | grad R² (elem) | grad R² (bfp) | bfp grad_survive | weight R² (bfp) |
|---|---|---|---|---|
| 5 | 0.9986 | 0.9986 | 0.9817 | 0.9701 |
| 4 | 0.9986 | 0.9986 | 0.9630 | 0.7672 |
| 3 | 0.9986 | 0.9986 | 0.9258 | 0.2784 |
| 2 | 0.9986 | 0.9986 | 0.8522 | 0.1246 |
| 1 | 0.9986 | 0.9986 | **0.7070** | 0.1629 |

Flat at 0.9986 at every width and in both formats, while `weight` collapses to
0.16. Widening the block to 512 pushes annihilation to 76% and R² still reads
0.9993.

**Transformer, AdamW, median of 3 seeds.** Paired excess perplexity over
FP32, with BFP gradient survival and KL against the same-seed FP32 model:

| bits | grad (elem) | grad (bfp) | grad_sr (bfp) | bfp grad_survive | bfp grad KL |
|---|---|---|---|---|---|
| 23 | -0.000 | -0.000 | +0.000 | 1.0000 | 0.000 |
| 7 | -0.001 | -0.001 | -0.008 | 0.9945 | 0.005 |
| 4 | +0.001 | -0.039 | -0.047 | 0.9562 | 0.054 |
| 2 | +0.013 | -0.006 | -0.012 | 0.8279 | 0.177 |
| 1 | -0.013 | -0.002 | +0.034 | **0.6820** | 0.233 |

Every median excess lies between -0.063 and +0.034 perplexity, inside the ±0.1
band that equally good runs scatter over. BFP annihilates 32% of gradient
elements at 1 bit. KL rises to 0.23, below the 0.29 to 0.33 that separates two
FP32 seeds (section 4), so the runs have reached a different, equally good
model rather than a worse one.

**Elementwise annihilates nothing, at any width, by construction.** Per-element
exponents mean the grid follows each value down, so no gradient can round to
zero. Only BFP's shared exponent can destroy one, which is why `grad_survive`
is 1.0000 down the whole elementwise column in both models.

#### Findings

**Stochastic rounding makes no difference.** On the CNN, KL(`grad`)/KL(`grad_sr`)
oscillates between 0.59 and 1.87 with no direction. On the MLP every
R²(`grad_sr`) − R²(`grad`) difference is within ±0.0001. On the transformer the
two differ by less than 0.06 perplexity at every width, in no consistent
direction. The control found nothing to control.

**AdamW does not change the answer.** Adam divides each gradient by a running
RMS of its own history, which could in principle amplify or hide rounding; the
transformer was the check, and it agrees with both SGD models.

**A refuted hypothesis, recorded because it was specific.** The first
explanation offered for the CNN result was that update-vanishing needs
*persistent* state: a weight is re-rounded every step so a discarded update is
gone for good, whereas a CNN gradient is recomputed from a fresh minibatch each
step and minibatch noise already dithers it. That predicts gradient
quantization should bite on the **full-batch** MLP, where the gradient is a
deterministic function of fixed data. It does not. Both regimes are free, at
comparable and higher annihilation. Batching is not the variable, and the
batch-size study proposed to isolate it is not worth running for this question.

**What survives is narrower and better evidenced.** Only the precision of the
stored, updated parameter matters. Coarsening the weight itself confines the
parameter to a grid permanently and erases sub-step updates; perturbing a
computation — input, activation, gradient, or the forward view of the weights
under `weight_master` — leaves the parameter free and a later gradient can
still move it. An annihilated gradient component is a missed step, not a lost
degree of freedom. This also explains `weight_master` without a separate
argument.

**`grad_survive` and `grad_cos` are new columns**, because `upd_survive` does
not transfer: under the grad conditions the weights stay FP32 and always move,
so it reads 1.0 while the gradient is being annihilated. `grad_cos` is reported
but must **not** be used to rank `grad` against `grad_sr` — it is a single-step
measure, and stochastic rounding buys unbiasedness with variance, so it
necessarily scores worse on any one draw while being no worse across a run.

**Gradients are outlier-structured on the CNN and Gaussian on the MLP.** Per-block
headroom p99 is 4–8 bits on CNN gradients against 2–3 for its weights, and it
*rises* through training; the MLP's gradients sit at 2.0–3.7, indistinguishable
from the iid Gaussian reference of 3.0. The structure also runs opposite to
activations — worst in the last layer for gradients, worst in the first for
activations — since activations carry the input's outlier structure forward and
gradients carry the loss's backward. This is why BFP annihilates far more
gradient on the CNN than the MLP at matched block size, and it is a real
difference the null result had to be checked against rather than assumed away.

**Correction to an earlier claim.** A previous version of this README asserted
that gradients reach the 1e-38 range and so make the subnormal-flushing decision
load-bearing. Measured, they do not: the smallest nonzero gradient observed was
2.8e-13 and **0.00%** of elements were subnormal at any layer or step. The
simulator keeps FP32 exponents, so underflow is an FP16 problem this design does
not have. Subnormal flushing stays cosmetic.

**The MLP under BFP.** Input quantization costs far more under BFP than
per-element exponents: R² 0.9558 against 0.9893 at 1 bit, a median loss ratio
of 31.7x against 7.8x. That is the same outlier-sensitivity story as the CNN's
activations. (An earlier version of this README gave 0.9562 and 47x from a
probe run that no committed file reproduces; these figures are from
`train_at_vary_precision.csv`.)

## Limitations

- **Optimizer state is never quantized.** One of the four categories in the premise remains unmeasured. It cannot be done on the CNN or MLP as configured: both run plain SGD at momentum 0, which carries no state. That experiment needs the transformer's AdamW, where stochastic rounding is also most likely to matter.
- **The gradient null result is bounded by what was tested.** Three models, two optimizers (SGD and AdamW), three and ten seeds. The transformer has LayerNorm, but LayerNorm parameters are skipped by the quantizers, so their gradients were never rounded; the weight gradients that flow through normalization were.
- **Architecture is not isolated anywhere.** The MLP is full-batch SGD, the CNN minibatch SGD, the transformer AdamW. The MLP is also the only model with a cliff. Because batching and optimizer covary with architecture, "quantization tolerance is architecture-specific" cannot be claimed from this data. Isolating it requires holding the optimizer and batching fixed across models. The one place batching *has* been compared directly is the gradient result in section 7, where full-batch and minibatch agree. The cliff itself is now known to be update-vanishing, since `weight_master` removes it.
- **Storage precision only.** All arithmetic is still done in FP32, and no narrow accumulator is simulated.
- **No normalization layers in the CNN**, which is the biggest risk to the outlier finding: normalization resets activation dynamic range every layer, and every production CNN has BatchNorm.
- **Accuracy saturates.** MNIST leaves only 1.3 points of headroom above the FP32 baseline, so nothing above 7 bits is resolvable by accuracy or loss. The reference-based metrics were added for this reason and should be read first on the CNN.
- **On the transformer the reference metrics saturate instead.** Two FP32 seeds already differ by KL 0.29 to 0.33, so KL against the same-seed reference cannot separate damage from divergence below that. That calibration was measured outside the sweep and is not yet a committed file.
- **Reproducibility floor is ±0.005 accuracy** from cuDNN nondeterminism, on 3 seeds. There is no `--deterministic` option yet. Setting `torch.backends.cudnn.deterministic` makes the CNN bit-reproducible. It does not do that for the transformer, whose FP32 training still jitters at about 1e-7 between identical runs; its `activation` runs varied by about 0.2 perplexity between identical runs while the logits were still rounded, and have not been re-measured since.
- **Fixed LR with no decay.** Production LR decay shrinks late-training updates, so this setup probably *understates* update-vanishing.
- **Activation hook sets differ slightly across models.** Neither rounds its output layer any more, but the transformer also rounds LayerNorm outputs, which the CNN does not have.
- The MLP study reports training loss only, with nothing held out. The CNN and transformer have proper validation splits; the CNN's test set is still untouched.
- Only 3 seeds on the CNN and transformer, against 10 on the MLP.
- BFP blocks are consecutive runs of the flattened tensor, not DYNASTY's 4x4 tiles.
- `--act-at consumer` has never been run, so the producer/consumer gap is unmeasured.

## Next steps

1. **Quantize optimizer state**, the last unmeasured category. This has to happen on the transformer: the CNN and MLP run SGD at momentum 0 and have no state to quantize. Adam's `exp_avg` and `exp_avg_sq` are persistent, updated tensors, which by the section 7 reading puts them in the *weight* class rather than the gradient class — so unlike gradients they should be genuinely sensitive, and stochastic rounding should matter there for the reason it did not matter for gradients. That is a sharp, falsifiable prediction and the highest-value experiment left.
2. **LR sweep at fixed bit width.** The mechanism says the collapse point is a race between gradient magnitude and grid spacing, so halving the LR should shift the collapse by one bit. This converts the update-vanishing story from an explanation into a falsifiable prediction, and it is the highest-value experiment currently possible with existing code.
3. **Commit the transformer's cross-seed calibration** as a sweep mode with its own results file, so the KL ceiling that section 4 relies on is reproducible from the repo rather than from a one-off script.
4. **Transformer PTQ.** The CNN's `--ptq` check has no transformer counterpart yet.
5. **DYNASTY itself:** Eq. 3b relative sensitivity, Algorithm 1 lambda tuning, EMA smoothing. Establish the equal-precision 8-bit BFP baseline first, since that is the paper's own comparison point.
6. Loose ends: raise the CNN and transformer to 10 seeds, rerun the CNN from a clean commit (its main sweep predates run metadata), run `--act-at consumer`, add a bits-per-element column, rename the `bfp16` tag to `bfp_b16`, retire `train_at_vary_precision_weight_grad_grad_sr.csv` now that the main MLP file reproduces it, and revisit the K=1024 accumulator anomaly. The `--batch-study` is no longer a priority: it was written to isolate gradient noise, and section 7 has now compared the two batching regimes directly.

**Protocol for every sweep from here.** Fix the epoch budget from the FP32 run and reuse it at every bit width; early stopping per run conflates "precision hurt the model" with "it trained for fewer epochs." Fix the validation split independently of the run seed. Log full curves every epoch and report final and best-epoch numbers side by side. Pair every reference-based metric within seed. Keep the test set untouched until the end.

## Open questions for the project

- **Block geometry, blocking.** `round_bfp` blocks along consecutive runs of the flattened tensor; DYNASTY uses 4x4 2D tiles. On the CNN's actual tensors this is about half clean: `linear` (10, 1568) gives blocks of 16 consecutive input features for one output neuron, but `conv1` (16, 1, 3, 3) and `conv2` (32, 16, 3, 3) have 9-element kernels, so each 16-block spans 1.78 kernels and straddles filter boundaries. Every CNN sweep run before this is settled would have to be discarded if it changes.
- What counts as reproducing the paper: equal-precision block floating point, or full DYNASTY?
- Are narrow accumulators in scope?
- Is MNIST plus a small transformer acceptable, or is CIFAR-100 with ResNet-18 required?
- Should normalization-layer weights be quantized? They are currently skipped, which matters now that the transformer has landed.
- Given that architecture is confounded with optimizer and batching, is isolating it worth the runs, or is the per-model tolerance question better answered by adding models than by controlling the confound?
