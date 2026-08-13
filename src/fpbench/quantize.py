"""Simulated low-precision formats, stored in FP32.

round_mantissa(x, bits): per-element exponent, `bits` mantissa bits.
round_bfp(x, bits, block): one shared exponent per block of elements.

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


def round_mantissa(x, bits):
    """Round mantissa to `bits` using round-half-to-even with per-element exponents."""
    if bits >= FP32_MANTISSA_BITS:
        return x.clone()

    u    = _bits(x)
    sign = u & SIGN_MASK             # Split off sign to prevent arithmetic shift smearing.
    mag  = u & MAG_MASK              
    drop = FP32_MANTISSA_BITS - bits
    lsb  = (mag >> drop) & 1         # Add LSB to round-half-to-even and prevent truncation.
    bias = (1 << (drop - 1)) - 1

    mag = ((mag + bias + lsb) >> drop) << drop
    return (mag | sign).contiguous().view(torch.float32)


def round_bfp(x, bits, block=16):
    """Apply Block Floating Point (BFP) quantization with one shared exponent per `block` of consecutive elements.

    Elements `g` exponents below the block maximum lose `g` mantissa bits, vanishing entirely if `g > bits`.
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
    q = torch.round(torch.ldexp(blk, -k))        # Scale exactly using ldexp to avoid underflow.
    out = torch.ldexp(q, k)

    out = torch.where(emax == 0, blk, out)       # Restore all-zero blocks.
    return out.reshape(-1)[:x.numel()].reshape(shape)


@torch.no_grad()
def quantize_weights(model, bits, block=None):
    """Quantize all weight matrices in place. Use `block=None` for per-element exponents.

    Biases and normalization parameters are explicitly skipped.
    """
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