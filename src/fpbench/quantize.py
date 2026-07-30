import torch

tiny = torch.finfo(torch.float32).tiny

def round_mantissa(x, bits):
    """
    Round x to the number of bits given
    Still stored as FP32
    """
    if bits >= 23:
        return x.clone()
    x_fl = x.float()
    exp = torch.floor(torch.log2(x_fl.abs().clamp(min = tiny)))
    scale = torch.exp2(exp - bits)
    out = torch.round(x_fl / scale) * scale
    return torch.where(x_fl == 0, x_fl, out)

def quantize_weights(model, bits):
    """
    Round every weight matrix in place. 
    Biases are untouched.
    """
    if bits >= 23:
        return
    with torch.no_grad(): #indicates not a trainable operation
        for name, p in model.named_parameters():
            if "weight" in name:
                p.copy_(round_mantissa(p, bits))