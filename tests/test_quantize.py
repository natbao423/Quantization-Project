import torch
from fpbench.quantize import round_mantissa

def test_matches_bf16():
    x = torch.randn(10000)
    assert torch.equal(round_mantissa(x, 7), x.bfloat16().float())

def test_matches_fp16():
    x = torch.randn(10000)
    assert torch.equal(round_mantissa(x, 10), x.half().float())