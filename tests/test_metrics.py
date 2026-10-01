"""Reference metrics (fpbench.metrics) and the transformer's fixed validation set."""

import pathlib
import sys

import pytest
import torch

from fpbench.metrics import logit_metrics

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_char_transformer as T      # noqa: E402


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(500, 65) * 3


def test_identical_logits_score_zero(logits):
    m = logit_metrics(logits, logits.clone())
    assert m == {"kl": 0.0, "disagree": 0.0, "logit_rel_err": 0.0,
                 "logit_rel_err_c": 0.0}


def test_a_per_row_shift_is_invisible_to_everything_but_the_uncentered_error(logits):
    """Softmax ignores a constant added to a row. Only logit_rel_err sees it,
    which is why the centered figure is the one to report."""
    shifted = logits + torch.randn(500, 1) * 10
    m = logit_metrics(shifted, logits)
    assert m["kl"] == pytest.approx(0, abs=1e-5)
    assert m["disagree"] == 0
    assert m["logit_rel_err_c"] == pytest.approx(0, abs=1e-5)
    assert m["logit_rel_err"] > 1


def test_kl_is_reference_first(logits):
    """KL(P_ref || P_q): mass the reference has and the run lost is what counts."""
    q = logits + torch.randn_like(logits)
    p_ref, p_q = logits.log_softmax(1), q.log_softmax(1)
    expect = (p_ref.exp() * (p_ref - p_q)).sum(1).mean().item()
    assert logit_metrics(q, logits)["kl"] == pytest.approx(expect, rel=1e-5)


def test_disagreement_counts_argmax_flips(logits):
    q = logits.clone()
    q[:50] = -q[:50]            # flips the argmax of the first 50 rows
    assert logit_metrics(q, logits)["disagree"] == pytest.approx(0.1)


def test_val_windows_tile_the_split_without_overlap():
    ids = torch.arange(1000)
    x, y = T.val_windows(ids)
    n = 999 // T.BLOCK_SIZE
    assert x.shape == y.shape == (n, T.BLOCK_SIZE)
    # Consecutive, non-overlapping, and every target is the next character.
    assert torch.equal(x.flatten(), ids[:n * T.BLOCK_SIZE])
    assert torch.equal(y.flatten(), ids[1:n * T.BLOCK_SIZE + 1])
