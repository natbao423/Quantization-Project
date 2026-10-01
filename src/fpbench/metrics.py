"""Metrics that compare a quantized run against an FP32 reference.

Accuracy and loss saturate: they say how good a model is, not how far
quantization moved it. These measure the perturbation directly, against the
FP32 model trained at the SAME SEED, so the initialization spread between seeds
is not folded into every number.

    kl              mean KL(FP32 || quantized) over the softmax outputs. Zero at
                    FP32, no ceiling, uses the whole distribution.
    disagree        fraction of rows whose argmax differs from the reference's.
    logit_rel_err   ||L_q - L_fp32|| / ||L_fp32|| over the whole set.
    logit_rel_err_c the same after removing each row's mean logit. Softmax
                    ignores a constant added to a row, so only this one predicts
                    KL; the gap between the two is the harmless shift.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def logit_metrics(logits, ref):
    """Compare a run's logits against the FP32 reference for the same seed.

    `logits` and `ref` are (rows, classes): one row per image on the CNN, one
    per predicted token on the transformer. Rows must line up one to one,
    so both have to come from the same fixed evaluation set in the same order.

    KL runs FP32-first, KL(P_fp32 || P_quant): it asks what the quantized model
    loses where the reference put its mass, which is the direction that
    punishes a confident wrong answer. The reverse direction would reward the
    quantized model for collapsing onto one class.
    """
    ref_logp = nn.functional.log_softmax(ref, dim=1)
    q_logp = nn.functional.log_softmax(logits, dim=1)
    # KL is nonnegative in exact arithmetic; clamped because identical
    # distributions land a few ulps below zero and a negative divergence in a
    # results table reads as a bug.
    kl = (ref_logp.exp() * (ref_logp - q_logp)).sum(1).mean().clamp_min(0)

    diff = logits - ref
    # Softmax ignores a constant added to every logit in a row, so the centered
    # figure is the part of the error that can change a prediction.
    dc = diff - diff.mean(1, keepdim=True)
    rc = ref - ref.mean(1, keepdim=True)

    return {
        "kl": kl.item(),
        "disagree": (logits.argmax(1) != ref.argmax(1)).float().mean().item(),
        "logit_rel_err": (diff.norm() / ref.norm()).item(),
        "logit_rel_err_c": (dc.norm() / rc.norm()).item(),
    }
