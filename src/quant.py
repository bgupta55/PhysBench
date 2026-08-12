import torch

def quantize_tensor(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax if w.abs().max() > 0 else 1.0
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax).to(torch.int32)
    return q, scale

def dequantize_tensor(q, scale):
    return q.to(torch.float32) * scale

def flip_bit(q_val: int, bit_pos: int, bits=8):
    mask = 1 << bit_pos
    biased = q_val + (1 << (bits - 1))
    biased &= (1 << bits) - 1
    biased ^= mask
    return biased - (1 << (bits - 1))
