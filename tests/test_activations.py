"""Activation rounding through forward hooks (fpbench.activations)."""

import torch

from fpbench.activations import QuantizedActivations
from fpbench.quantize import quantize


def _two_layers():
    torch.manual_seed(0)
    body, head = torch.nn.Linear(8, 8), torch.nn.Linear(8, 5)
    return torch.nn.Sequential(body, head), body, head


def test_every_matching_module_is_rounded_by_default():
    model, body, head = _two_layers()
    x = torch.randn(4, 8)
    with torch.no_grad(), QuantizedActivations(model, 2, types=(torch.nn.Linear,)):
        got = model(x)
        expect = quantize(head(quantize(body(x), 2)), 2)
    assert torch.equal(got, expect)


def test_an_excluded_module_is_left_alone_and_the_rest_still_rounded():
    """The transformer's head: its input is rounded, its output (the logits)
    is not."""
    model, body, head = _two_layers()
    x = torch.randn(4, 8)
    with torch.no_grad():
        with QuantizedActivations(model, 2, types=(torch.nn.Linear,), exclude=(head,)):
            got = model(x)
        expect = head(quantize(body(x), 2))
    assert torch.equal(got, expect)
    assert not torch.equal(got, quantize(got, 2))     # the logits really are unrounded


def test_hooks_are_removed_on_exit():
    model, body, head = _two_layers()
    with QuantizedActivations(model, 2, types=(torch.nn.Linear,), exclude=(head,)):
        assert len(body._forward_hooks) == 1 and len(head._forward_hooks) == 0
    assert len(body._forward_hooks) == 0
