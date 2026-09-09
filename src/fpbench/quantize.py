"""Simulated low-precision formats, stored in FP32.

round_mantissa(x, bits): per-element exponent, `bits` mantissa bits.
round_bfp(x, bits, block): one shared exponent per block of elements.

Both take `stochastic=True`, which replaces round-to-nearest with an unbiased
random dither. Round-to-nearest is deterministic, so a value sitting below half
a grid step rounds down every time and its contribution is discarded for good;
stochastic rounding makes it survive in proportion to its size, turning a
discard into a deferral. That costs variance -- a single stochastic draw is
further from the original than round-to-nearest -- so the benefit is only
visible across many steps, never in a one-shot comparison.

Both return FP32. Stored values are constrained to the target grid, leaving subsequent arithmetic unaffected.

Design notes
------------
Rounding uses exact integer bit manipulation. Subnormals are flushed to signed zero, accurately reflecting real BFP hardware.

The round_bfp function allows carry-out: block exponents increment if the largest element rounds up across a power of two. This guarantees block=1 reduces exactly to round_mantissa. As a deliberate format property, round_bfp is not strictly idempotent.

Inf and NaN inputs are not handled.
"""

import torch

SIGN_MASK = -2147483648          # 0x80000000 as a signed int32
EXP_MASK  = 0x7F800000
MAG_MASK  = 0x7FFFFFFF
FP32_MANTISSA_BITS = 23
FP32_EXP_BIAS = 127


def _bits(x):
    """Reinterpret a float32 tensor as int32 without changing bits."""
    return x.contiguous().view(torch.int32)


def flush_subnormals(x):
    """Map subnormal inputs to signed zero while preserving the original sign."""
    u = _bits(x)
    return torch.where((u & EXP_MASK) != 0, u, u & SIGN_MASK) \
                .contiguous().view(torch.float32)


def round_mantissa(x, bits, *, stochastic=False, generator=None):
    """Round mantissa to `bits` with per-element exponents.

    Default is round-half-to-even. With `stochastic=True` the fixed rounding
    bias is replaced by a uniform random dither, which makes the operation
    unbiased: an element whose dropped bits are a fraction f of a grid step
    rounds up with probability exactly f, so E[round(x)] == x.

    Why that matters for gradients. Round-to-nearest is deterministic, so a
    value that sits systematically below half a step rounds the same way every
    step and its contribution is discarded permanently -- the same
    update-vanishing that separates `weight` from `weight_master`. Stochastic
    rounding converts that discard into a deferral: the contribution lands
    eventually, in proportion to its size. The cost is variance, so a single
    rounded tensor is FURTHER from the original than round-to-nearest, not
    closer. The benefit only appears across many steps, and any single-step
    metric (cosine similarity to the FP32 tensor, say) will rank it worse.

    `generator` seeds the dither. Left as None it draws from the global RNG,
    which every sweep already seeds per run via torch.manual_seed, so runs stay
    reproducible without threading a generator through; pass one explicitly to
    isolate the stream. A generator must live on the same device as `x` --
    a CPU torch.Generator against a CUDA tensor raises. The sweeps shuffle
    batches with a CPU generator, so do not reuse that one here.
    """
    if bits >= FP32_MANTISSA_BITS:
        return x.clone()

    u    = _bits(x)
    sign = u & SIGN_MASK             # Split off sign to prevent arithmetic shift smearing.
    mag  = u & MAG_MASK              
    drop = FP32_MANTISSA_BITS - bits

    if stochastic:
        # Uniform in [0, 2^drop). The carry out of the dropped field happens
        # exactly when dither >= 2^drop - f, so P(round up) = f / 2^drop.
        dither = torch.randint(0, 1 << drop, x.shape, dtype=torch.int32,
                               device=x.device, generator=generator)
    else:
        lsb  = (mag >> drop) & 1     # Add LSB to round-half-to-even and prevent truncation.
        dither = (1 << (drop - 1)) - 1 + lsb

    mag = ((mag + dither) >> drop) << drop
    return (mag | sign).contiguous().view(torch.float32)


def round_bfp(x, bits, block=16, *, stochastic=False, generator=None):
    """Apply Block Floating Point (BFP) quantization with one shared exponent per `block` of consecutive elements.

    Elements `g` exponents below the block maximum lose `g` mantissa bits.

    They do not vanish until `g >= bits + 2`. The block's grid spacing is
    2^(emax - bits), and rounding is to nearest, so an element survives until it
    drops below HALF a step: 2^-g < 2^-(bits+1) gives g > bits + 1. Exact ties
    at g == bits + 1 also vanish, because half-to-even rounds them to zero.
    Verified at 4 bits: 1.3*2^-5 survives, 1.3*2^-6 is the first to vanish, and
    the exact power of two 2^-5 vanishes one exponent early as a tie.

    NOT a no-op at bits=23. Unlike round_mantissa, where 23 mantissa bits is
    exactly FP32, a shared exponent still forces sub-max elements onto a coarser
    grid (measured: 367 of 4608 elements change on a (32,16,3,3) conv weight).
    Callers that use 23 as a sentinel for "quantization off" must guard it
    themselves; quantize_weights does.

    `stochastic=True` swaps round-to-nearest for an unbiased random dither; see
    round_mantissa for what that buys and what it costs. Unlike the
    round-to-nearest path, `block=1` stochastic is only distributionally equal
    to stochastic round_mantissa, not bit-identical, since the two draw
    different random streams.
    """
    shape = x.shape
    flat  = flush_subnormals(x).reshape(-1)

    # Blocks are consecutive flattened elements. 
    # In Linear(out, in) weights, this groups inputs per output neuron, sharing exponents across terms summed in matmuls.
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.view(-1, block)

    efield = (_bits(blk) >> FP32_MANTISSA_BITS) & 0xFF   # 0 indicates a zero element.
    emax   = efield.max(dim=1, keepdim=True).values

    k = emax - FP32_EXP_BIAS - bits              # Set shared step size.
    scaled = torch.ldexp(blk, -k)                # Scale exactly using ldexp to avoid underflow.
    if stochastic:
        # floor(s + U) rounds up with probability equal to s's fractional part,
        # for negative s as well as positive. See round_mantissa on why this is
        # worth the added variance.
        u = torch.rand(scaled.shape, dtype=scaled.dtype, device=scaled.device,
                       generator=generator)
        q = torch.floor(scaled + u)
    else:
        q = torch.round(scaled)
    out = torch.ldexp(q, k)

    out = torch.where(emax == 0, blk, out)       # Restore all-zero blocks.
    return out.reshape(-1)[:x.numel()].reshape(shape)


@torch.no_grad()
def quantize_weights(model, bits, block=None):
    """Quantize all weight matrices in place. Use `block=None` for per-element exponents.

    Biases and normalization parameters are explicitly skipped.

    `bits >= 23` means "quantization off" and touches nothing. The guard has to
    live here rather than in round_bfp: BFP-23 is a real format that genuinely
    coarsens sub-max elements, so making the primitive lie about it would
    corrupt the one function the test suite validates bit-exactly. It is only
    the SWEEPS that overload 23 as a sentinel, and the sweeps enter through
    here. Without this, the bfp16 23-bit row quantized weights in the `weight`,
    `both` and `act_weight` conditions while `weight_master` (which guards in
    QuantizedForward) and `input`/`activation` did not, so the six conditions
    that are supposed to share one FP32 baseline did not actually share it.
    """
    if bits >= FP32_MANTISSA_BITS:
        return

    # Skip normalization layers to avoid conflating numerical effects on standalone scale parameters.
    skip = (torch.nn.LayerNorm, torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d, torch.nn.GroupNorm)

    for mod in model.modules():
        if isinstance(mod, skip):
            continue
        w = getattr(mod, "weight", None)
        if w is None or not torch.is_floating_point(w):
            continue
        q = round_mantissa(w, bits) if block is None else round_bfp(w, bits, block)
        w.copy_(q)


def block_exponent_stats(x, block=16):
    """Per-block exponent statistics in bits: (spread, headroom).
    spread   = emax - emin     predicts BFP damage; how many bits the smallest
                               element loses to the alignment shift
    headroom = emax - emedian  detects outliers; insensitive to the near-zero
                               tail that dominates spread for any continuous
                               distribution
    """
    flat = flush_subnormals(x).reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    e = ((_bits(flat.view(-1, block)) >> FP32_MANTISSA_BITS) & 0xFF).float()
    e[e == 0] = float("nan") #ignores zeros                       
    valid = ~e.isnan().all(dim=1)
    emax = e.nan_to_num(nan=0.0).amax(dim=1)
    emin = e.nan_to_num(nan=255.0).amin(dim=1)
    emed = e.nanmedian(dim=1).values
    return (emax - emin)[valid], (emax - emed)[valid]