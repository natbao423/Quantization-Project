"""Activation quantization and outlier diagnostics.

Weights are quantized by rounding them in place after each optimizer step.
Activations cannot be: they are produced fresh in every forward pass and sit
inside the autograd graph. Both problems are handled here.

Two context managers, both non-invasive: they attach forward hooks on entry and
remove them on exit, so the model definition never mentions quantization.
"""

import torch
import torch.nn as nn

from fpbench.quantize import round_mantissa, round_bfp, block_exponent_stats

DEFAULT_TYPES = (nn.Linear, nn.LayerNorm)


def quantize(x, bits, block=None):
    """block=None gives per-element exponents; block=N gives BFP."""
    if bits >= 23:
        return x
    return round_mantissa(x, bits) if block is None else round_bfp(x, bits, block)


def quantize_ste(x, bits, block=None):
    """Quantize forward, pass the gradient through unchanged (straight-through).

    Rounding has zero derivative almost everywhere, so a faithful backward pass
    would kill training outright. The standard workaround is to pretend the
    rounding was the identity when computing gradients. The forward value is
    exactly the quantized one; only the backward path is a fiction.

    This is a different mechanism from the weight quantization used elsewhere,
    which rounds in place under no_grad and has no gradient path at all.
    """
    if bits >= 23:
        return x
    q = quantize(x.detach(), bits, block)
    return x + (q - x).detach()


class QuantizedActivations:
    """Round the output of every matching submodule during the forward pass.

        with QuantizedActivations(model, bits=4, block=16):
            loss = criterion(model(x), y)
    """

    def __init__(self, model, bits, block=None, types=DEFAULT_TYPES):
        self.model, self.bits, self.block, self.types = model, bits, block, types
        self.handles = []

    def __enter__(self):
        if self.bits >= 23:
            return self

        def hook(_module, _inputs, output):
            if torch.is_tensor(output) and torch.is_floating_point(output):
                return quantize_ste(output, self.bits, self.block)
            return output

        for m in self.model.modules():
            if isinstance(m, self.types):
                self.handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


class ActivationStats:
    """Per-block exponent statistics for every matching submodule's output.

    spread   = emax - emin,      predicts BFP damage
    headroom = emax - emedian,   detects outliers

    Measured reference values for 16 iid samples, so a model can be compared
    against distributions with known outlier structure:

        distribution                 spread med/p99   headroom med/p99
        iid uniform(-1,1)              4.0 / 10.0        0.5 /  1.5
        iid Gaussian                   5.0 / 11.0        2.0 /  3.0
        heavy tail (t, df=3)           6.0 / 12.0        2.0 /  4.5
        Gaussian + 1% outliers x10     5.0 / 12.0        2.0 /  5.0
        Gaussian + 1% outliers x100    6.0 / 14.0        2.0 /  8.0

    Spread cannot discriminate between these: it is dominated by the smallest
    element in the block, which lands near zero for any continuous
    distribution. Headroom can.
    """

    def __init__(self, model, block=16, types=DEFAULT_TYPES):
        self.model, self.block, self.types = model, block, types
        self.stats = {}
        self.handles = []

    def __enter__(self):
        def make_hook(name):
            @torch.no_grad()
            def hook(_module, _inputs, output):
                if not (torch.is_tensor(output) and torch.is_floating_point(output)):
                    return
                spread, head = block_exponent_stats(output.detach(), self.block)
                keep = torch.isfinite(spread) & torch.isfinite(head)
                if keep.any():
                    s, h = self.stats.setdefault(name, ([], []))
                    s.append(spread[keep].float().cpu())
                    h.append(head[keep].float().cpu())
            return hook

        for name, m in self.model.named_modules():
            if isinstance(m, self.types):
                self.handles.append(m.register_forward_hook(make_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False

    def summary(self):
        """name -> spread and headroom medians, p99s, and the fraction of
        blocks whose spread exceeds 8 bits (which 8-bit BFP cannot represent:
        the smallest element shifts out of the mantissa before rounding)."""
        out = {}
        for name, (schunks, hchunks) in self.stats.items():
            s, h = torch.cat(schunks), torch.cat(hchunks)
            out[name] = {
                "spread_med": s.median().item(),
                "spread_p99": s.quantile(0.99).item(),
                "head_med": h.median().item(),
                "head_p99": h.quantile(0.99).item(),
                "frac_over_8": (s > 8).float().mean().item(),
                "n_blocks": s.numel(),
            }
        return out

    def print_summary(self, title=""):
        rows = self.summary()
        if title:
            print(f"\n{title}")
        print(f"{'module':26s} {'spr med':>8} {'spr p99':>8} "
              f"{'hd med':>8} {'hd p99':>8} {'>8 bits':>9}")
        for name, s in rows.items():
            print(f"{name:26s} {s['spread_med']:8.2f} {s['spread_p99']:8.2f} "
                  f"{s['head_med']:8.2f} {s['head_p99']:8.2f} "
                  f"{s['frac_over_8']:8.1%}")
        print(f"{'(iid Gaussian reference)':26s} {5.0:8.2f} {11.0:8.2f} "
              f"{2.0:8.2f} {3.0:8.2f} {0.072:8.1%}")
        return rows