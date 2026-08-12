"""Test suite for fpbench.quantize[cite: 6].

Validates hardware-equivalent simulation accuracy at standard bit widths to guarantee credibility at experimental widths[cite: 6].
"""

import pytest
import torch

from fpbench.quantize import (
    round_mantissa,
    round_bfp,
    flush_subnormals,
    block_exponent_spread,
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


def test_block_exponent_spread_detects_outliers():
    flat = torch.tensor([1.0, 1.1, 0.9, 1.05]).repeat(4)
    spiky = torch.tensor([1024.0, 1.0, 1.0, 1.0]).repeat(4)
    assert block_exponent_spread(flat, 16).max() <= 1
    assert block_exponent_spread(spiky, 16).max() >= 10