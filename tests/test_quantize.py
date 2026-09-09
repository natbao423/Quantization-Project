"""Test suite for fpbench.quantize[cite: 6].

Validates hardware-equivalent simulation accuracy at standard bit widths to guarantee credibility at experimental widths[cite: 6].
"""

import pytest
import torch

from fpbench.quantize import (
    round_mantissa,
    round_bfp,
    flush_subnormals,
    block_exponent_stats,
    quantize_weights,
)

BITS = [0, 1, 3, 5, 7, 10, 17]


def mantissa_low_bits(x, bits):
    """Extract mantissa bits that `bits` quantization guarantees to clear[cite: 6]."""
    u = x.contiguous().view(torch.int32)
    return u & ((1 << (23 - bits)) - 1)


@pytest.fixture
def gaussian():
    torch.manual_seed(0)
    return torch.randn(200_000)


# --------------------------------------------------------------------------
# 1. the bit-level test from the meeting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bits", BITS)
def test_low_mantissa_bits_are_exactly_zero(gaussian, bits):
    """Verify exact zeroing of the lowest 6 mantissa bits when reducing from 23 to 17 bits[cite: 6]."""
    q = round_mantissa(gaussian, bits)
    assert torch.count_nonzero(mantissa_low_bits(q, bits)) == 0


@pytest.mark.parametrize("bits", BITS)
def test_rounding_preserves_sign(gaussian, bits):
    q = round_mantissa(gaussian, bits)
    assert torch.equal(torch.signbit(q), torch.signbit(gaussian))


@pytest.mark.parametrize("bits", BITS)
def test_rounding_is_idempotent(gaussian, bits):
    q = round_mantissa(gaussian, bits)
    assert torch.equal(round_mantissa(q, bits), q)


def test_rounding_error_is_bounded(gaussian):
    """Ensure relative rounding error remains strictly bounded below half a grid step[cite: 6]."""
    for bits in BITS:
        q = round_mantissa(gaussian, bits)
        rel = (q - gaussian).abs() / gaussian.abs()
        assert rel.max() <= 2.0 ** (-bits - 1) * 1.0001


# --------------------------------------------------------------------------
# 2. hardware equivalence
# --------------------------------------------------------------------------

def test_matches_bfloat16(gaussian):
    assert torch.equal(round_mantissa(gaussian, 7), gaussian.bfloat16().float())


def test_matches_float16_in_normal_range(gaussian):
    """Verify exact float16 equivalence within the normal exponent range[cite: 6]. Subnormals are excluded[cite: 6]."""
    normal = gaussian.abs() >= torch.finfo(torch.float16).tiny
    assert torch.equal(
        round_mantissa(gaussian, 10)[normal],
        gaussian.half().float()[normal],
    )


def test_float16_carveout_is_small(gaussian):
    """Ensure subnormal carve-out proportion remains statistically negligible[cite: 6]."""
    below = (gaussian.abs() < torch.finfo(torch.float16).tiny).float().mean()
    assert below < 1e-4


# --------------------------------------------------------------------------
# 3. subnormals and edge cases
# --------------------------------------------------------------------------

def test_no_nan_on_subnormals():
    """Verify subnormal inputs resolve to finite values without generating NaNs[cite: 6]."""
    edge = torch.tensor([0.0, -0.0, 1e-38, -1e-40, 5e-44, -1.4e-45,
                         1.0, -1.0, -2.5])
    for bits in BITS:
        assert torch.isfinite(round_mantissa(edge, bits)).all()
        for block in (1, 4, 16):
            assert torch.isfinite(round_bfp(edge, bits, block)).all()


def test_flush_preserves_signed_zero():
    x = torch.tensor([0.0, -0.0, 1e-40, -1e-40])
    out = flush_subnormals(x)
    assert torch.equal(out, torch.zeros(4))
    assert torch.equal(torch.signbit(out), torch.signbit(x))


# --------------------------------------------------------------------------
# 4. block floating point
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [1, 3, 5, 7, 10, 17])
def test_bfp_block_one_reduces_to_round_mantissa(gaussian, bits):
    """Verify block=1 BFP reduces exactly to round_mantissa[cite: 6]."""
    assert torch.equal(
        round_bfp(gaussian, bits, block=1),
        round_mantissa(flush_subnormals(gaussian), bits),
    )


def test_bfp_leaves_a_shared_exponent_block_untouched():
    """Ensure uniform exponent blocks remain unchanged under sufficient bit width[cite: 6]."""
    x = torch.tensor([1.5, 1.75, -1.25, 1.0]).repeat(4)
    assert torch.equal(round_bfp(x, 10, block=16), x)


def test_bfp_outlier_crushes_its_block():
    """Verify outliers dictate block exponent scales and force smaller elements to zero[cite: 6]."""
    x = torch.tensor([1024.0, 0.5, 0.25, 0.125]).repeat(4)

    shared = round_bfp(x, 4, block=16)
    assert shared[0] == 1024.0
    assert torch.equal(shared[1:4], torch.zeros(3))

    per_element = round_bfp(x, 4, block=1)
    assert torch.equal(per_element[:4], x[:4])


@pytest.mark.parametrize("shape", [(7,), (32, 16), (8, 3, 3, 3), (5, 5)])
@pytest.mark.parametrize("block", [1, 4, 16, 32])
def test_bfp_preserves_shape_with_padding(shape, block):
    torch.manual_seed(1)
    w = torch.randn(shape)
    assert round_bfp(w, 6, block).shape == w.shape


def test_bfp_instability_is_confined_to_carried_blocks():
    """Guarantee lack of idempotency is strictly confined to blocks experiencing power-of-two carry-overs[cite: 6]."""
    torch.manual_seed(2)
    x = torch.randn(4096)
    a = round_bfp(x, 6, 16)
    b = round_bfp(a, 6, 16)

    def block_emax(t):
        u = t.contiguous().view(torch.int32).view(-1, 16)
        return ((u >> 23) & 0xFF).amax(dim=1)

    carried = block_emax(x) != block_emax(a)
    moved = (a != b).view(-1, 16).any(dim=1)
    assert torch.equal(moved, moved & carried)
    assert carried.float().mean() < 0.02      # measured ~0.77%


def test_headroom_detects_outliers_where_spread_cannot():
    """Spread is dominated by the smallest element in a block, which lands near
    zero for any continuous distribution, so it cannot tell an outlier-heavy
    tensor from a Gaussian one. Headroom compares the max against the block
    median and can."""
    torch.manual_seed(0)
    plain = torch.randn(16 * 20_000)
    spiky = plain.clone()
    spiky[::100] *= 100                       # 1% of elements, 100x larger

    s_plain, h_plain = block_exponent_stats(plain, 16)
    s_spiky, h_spiky = block_exponent_stats(spiky, 16)

    # headroom separates them clearly
    assert h_spiky.quantile(0.99) >= h_plain.quantile(0.99) + 3

    # spread barely moves, which is the reason headroom exists
    assert s_spiky.quantile(0.99) - s_plain.quantile(0.99) <= 4


def test_spread_measures_bfp_alignment_loss():
    """Spread is still the right statistic for its own purpose: how many bits
    the smallest element loses to the shared exponent."""
    flat = torch.tensor([1.0, 1.1, 0.9, 1.05]).repeat(4)
    wide = torch.tensor([1024.0, 1.0, 1.0, 1.0]).repeat(4)
    assert block_exponent_stats(flat, 16)[0].max() <= 1
    assert block_exponent_stats(wide, 16)[0].max() >= 10


# --------------------------------------------------------------------------
# 5. where sub-max elements actually vanish
# --------------------------------------------------------------------------

def _sub_max_block(g, scale=1.3):
    """A block whose max is 1.0 and whose second element sits `g` exponents down.

    scale=1.3 keeps the value off the exact half-step tie; scale=1.0 lands on it.
    """
    x = torch.zeros(16)
    x[0] = 1.0
    x[1] = scale * 2.0 ** -g
    return x


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 7])
def test_bfp_element_survives_until_half_a_step(bits):
    """The threshold is g >= bits + 2, not g > bits.

    Grid spacing inside the block is 2^(emax - bits) and rounding is to nearest,
    so an element survives until it falls below half a step. The docstring
    originally claimed one exponent earlier, which would have overstated BFP
    damage by a full bit everywhere the headroom diagnostic is read.
    """
    assert round_bfp(_sub_max_block(bits + 1), bits, 16)[1] != 0
    assert round_bfp(_sub_max_block(bits + 2), bits, 16)[1] == 0


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 7])
def test_bfp_exact_tie_vanishes_one_exponent_early(bits):
    """An element at exactly half a step is a tie, and half-to-even picks zero.

    This is the carve-out on the rule above, and the reason a naive check with
    powers of two alone reports the old (wrong) threshold.
    """
    assert round_bfp(_sub_max_block(bits + 1, scale=1.0), bits, 16)[1] == 0


# --------------------------------------------------------------------------
# 6. the 23-bit sentinel
# --------------------------------------------------------------------------

def test_round_bfp_is_not_a_no_op_at_23_bits():
    """BFP-23 is a real format, not FP32: the shared exponent still coarsens
    sub-max elements. This is why the sentinel guard belongs in
    quantize_weights rather than in the primitive."""
    torch.manual_seed(3)
    w = torch.randn(32, 16, 3, 3)             # conv2's shape in the CNN sweep
    assert not torch.equal(round_bfp(w, 23, 16), w)


@pytest.mark.parametrize("block", [None, 1, 16])
def test_quantize_weights_is_a_no_op_at_23_bits(block):
    """Every condition in a 23-bit sweep row must be the same FP32 baseline.

    round_mantissa gives this for free, since 23 mantissa bits is FP32.
    round_bfp does not, so without the guard the bfp16 23-bit row quantized
    weights in three of its six conditions and left the other three alone,
    while being reported as the noise floor for all of them.
    """
    torch.manual_seed(4)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.Linear(32 * 4, 10),
    )
    before = [p.detach().clone() for p in model.parameters()]
    quantize_weights(model, 23, block)
    assert all(torch.equal(a, b)
               for a, b in zip(before, model.parameters()))


@pytest.mark.parametrize("block", [None, 16])
def test_quantize_weights_still_quantizes_below_23_bits(block):
    """The guard must not swallow the real cases it sits in front of."""
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3, padding=1))
    before = model[0].weight.detach().clone()
    quantize_weights(model, 4, block)
    assert not torch.equal(before, model[0].weight.detach())


def test_quantize_weights_skips_normalization_and_bias():
    """Normalization scales and biases stay FP32 at every width."""
    torch.manual_seed(6)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.LayerNorm(16))
    norm_w = model[1].weight.detach().clone()
    bias = model[0].bias.detach().clone()
    quantize_weights(model, 1, 16)
    assert torch.equal(norm_w, model[1].weight.detach())
    assert torch.equal(bias, model[0].bias.detach())

# --------------------------------------------------------------------------
# 7. stochastic rounding
# --------------------------------------------------------------------------

def _expected(fn, n, seed=0):
    """Elementwise E[fn()] over n independent draws, accumulated in place.

    Bias is a PER-ELEMENT property: E[q_i] - x_i. Averaging the signed error
    over a symmetric tensor first is the wrong measure, because the
    round-to-nearest errors of positive and negative elements cancel and it
    looks unbiased when it is not.
    """
    g = torch.Generator().manual_seed(seed)
    total = None
    for _ in range(n):
        q = fn(g)
        total = q.double() if total is None else total + q.double()
    return (total / n).float()


@pytest.mark.parametrize("bits", [1, 3, 5, 7, 10])
def test_stochastic_lands_on_the_same_grid(gaussian, bits):
    """Unbiased does not mean off-grid: the output is still a representable
    value at `bits` mantissa bits, exactly as round-to-nearest is."""
    q = round_mantissa(gaussian, bits, stochastic=True)
    assert torch.count_nonzero(mantissa_low_bits(q, bits)) == 0


@pytest.mark.parametrize("bits", [1, 3, 7])
def test_stochastic_preserves_sign(gaussian, bits):
    q = round_mantissa(gaussian, bits, stochastic=True)
    assert torch.equal(torch.signbit(q), torch.signbit(gaussian))


def test_stochastic_preserves_exact_zeros():
    """A dither smaller than one step can never lift zero off zero. Matters
    because ReLU makes activations genuinely sparse."""
    z = torch.zeros(1000)
    for bits in [0, 1, 5, 10]:
        assert torch.equal(round_mantissa(z, bits, stochastic=True), z)


def test_stochastic_is_a_no_op_at_23_bits(gaussian):
    assert torch.equal(round_mantissa(gaussian, 23, stochastic=True), gaussian)


def test_stochastic_rounds_up_with_the_right_probability():
    """The exact claim: an element sitting a fraction f of a step above the
    grid point rounds up with probability f.

    1.125 at 1 mantissa bit sits a quarter step above 1.0, with 1.5 the next
    grid point up, so it must land on 1.5 a quarter of the time.
    """
    x = torch.full((40_000,), 1.125)
    g = torch.Generator().manual_seed(0)
    q = round_mantissa(x, 1, stochastic=True, generator=g)

    assert set(q.unique().tolist()) == {1.0, 1.5}
    up = (q == 1.5).float().mean().item()
    assert abs(up - 0.25) < 0.01
    assert abs(q.mean().item() - 1.125) < 0.005


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_stochastic_is_unbiased_and_round_to_nearest_is_not(gaussian, bits):
    """The whole point. Averaged over draws the stochastic error vanishes;
    the round-to-nearest error is a fixed offset that no amount of averaging
    removes, which is what lets it accumulate across training steps."""
    x = gaussian[:20_000]
    sr = _expected(lambda g: round_mantissa(x, bits, stochastic=True, generator=g), 200)
    rtn = round_mantissa(x, bits)

    # |E[q] - x| per element. Stochastic shrinks as 1/sqrt(draws); the
    # round-to-nearest offset is fixed and no averaging touches it.
    assert (sr - x).abs().mean() < (rtn - x).abs().mean() / 5


@pytest.mark.parametrize("bits", [1, 3, 7])
def test_stochastic_costs_variance_on_a_single_draw(gaussian, bits):
    """Locking in the counter-intuitive half: ONE stochastic draw is further
    from the original than round-to-nearest, because unbiasedness is bought
    with variance. Any single-step metric will therefore rank stochastic
    rounding worse, and that is not a bug to be fixed."""
    g = torch.Generator().manual_seed(0)
    sr = round_mantissa(gaussian, bits, stochastic=True, generator=g)
    rtn = round_mantissa(gaussian, bits)
    assert (sr - gaussian).abs().mean() > (rtn - gaussian).abs().mean()


@pytest.mark.parametrize("bits", [1, 3, 7])
def test_stochastic_error_is_bounded_by_one_full_step(gaussian, bits):
    """Round-to-nearest is bounded by half a step; stochastic by a whole one,
    since it may round away from the closer neighbour."""
    g = torch.Generator().manual_seed(0)
    q = round_mantissa(gaussian, bits, stochastic=True, generator=g)
    rel = (q - gaussian).abs() / gaussian.abs()
    assert rel.max() <= 2.0 ** (-bits) * 1.0001


def test_stochastic_is_reproducible_from_a_seed():
    """Sweeps must stay reproducible once a random dither is in the loop."""
    x = torch.randn(10_000)
    a = round_mantissa(x, 3, stochastic=True,
                       generator=torch.Generator().manual_seed(7))
    b = round_mantissa(x, 3, stochastic=True,
                       generator=torch.Generator().manual_seed(7))
    c = round_mantissa(x, 3, stochastic=True,
                       generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# --- block floating point ---------------------------------------------------

@pytest.mark.parametrize("bits", [1, 3, 5])
def test_bfp_stochastic_is_unbiased(bits):
    torch.manual_seed(0)
    x = torch.randn(16 * 1000)
    sr = _expected(lambda g: round_bfp(x, bits, 16, stochastic=True, generator=g), 200)
    rtn = round_bfp(x, bits, 16)
    assert (sr - x).abs().mean() < (rtn - x).abs().mean() / 5


def test_bfp_stochastic_leaves_all_zero_blocks_alone():
    """The emax == 0 passthrough must survive the dither, or padding and dead
    ReLU blocks would acquire noise."""
    x = torch.zeros(64)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(round_bfp(x, 3, 16, stochastic=True, generator=g), x)


def test_bfp_stochastic_block_one_matches_round_mantissa_in_distribution():
    """block=1 reduces to round_mantissa bit-exactly for round-to-nearest, but
    stochastically the two draw different random streams, so only their means
    agree. Documented rather than asserted as equality."""
    x = torch.full((40_000,), 1.125)
    ga = torch.Generator().manual_seed(0)
    gb = torch.Generator().manual_seed(1)
    a = round_bfp(x, 1, block=1, stochastic=True, generator=ga)
    b = round_mantissa(x, 1, stochastic=True, generator=gb)

    assert not torch.equal(a, b)
    assert set(a.unique().tolist()) == set(b.unique().tolist()) == {1.0, 1.5}
    assert abs(a.mean().item() - b.mean().item()) < 0.01


def test_bfp_stochastic_can_rescue_a_crushed_element():
    """The mechanism, in miniature. An element far below its block max rounds
    to zero every single time under round-to-nearest, so its contribution is
    gone for good. Stochastically it survives sometimes, in proportion to its
    size -- discarded becomes deferred."""
    x = torch.zeros(16)
    x[0] = 1.0
    x[1] = 2.0 ** -8            # 8 exponents down, far past the 4-bit cutoff

    assert round_bfp(x, 4, 16)[1] == 0

    g = torch.Generator().manual_seed(0)
    survived = sum(round_bfp(x, 4, 16, stochastic=True, generator=g)[1] != 0
                   for _ in range(2000))
    assert survived > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stochastic_rejects_a_generator_on_the_wrong_device():
    """Documents the contract. The sweeps shuffle batches with a CPU generator,
    so reusing it for a CUDA dither is an easy mistake; it must fail loudly
    rather than silently fall back to unseeded randomness."""
    x = torch.randn(1024, device="cuda")
    cpu_gen = torch.Generator().manual_seed(0)
    with pytest.raises(RuntimeError, match="device"):
        round_mantissa(x, 3, stochastic=True, generator=cpu_gen)
    with pytest.raises(RuntimeError, match="device"):
        round_bfp(x, 3, 16, stochastic=True, generator=cpu_gen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stochastic_is_reproducible_from_the_global_seed_on_cuda():
    """How the sweeps will actually use it: no explicit generator, seeded by
    the torch.manual_seed(seed) that run() already calls."""
    x = torch.randn(100_000, device="cuda")
    torch.manual_seed(0); a = round_bfp(x, 3, 16, stochastic=True)
    torch.manual_seed(0); b = round_bfp(x, 3, 16, stochastic=True)
    torch.manual_seed(1); c = round_bfp(x, 3, 16, stochastic=True)
    assert torch.equal(a, b) and not torch.equal(a, c)
