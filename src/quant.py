import torch
import laq_cuda

window_size = 32

def per_channel_quant(t):
    # quantization
    B, H, S, C = t.shape
    o_S = S
    x = t
    if S % window_size != 0:
        # residual tensor
        S = S - S % window_size
        x = t[:, :, :S,:]
    out_q = torch.empty(B, H, S, C, dtype=torch.int8, device='cuda')
    out_s = torch.empty((S // window_size, C), dtype=torch.float32, device='cuda')
    laq_cuda.quant(x, out_q, out_s, B, H, S, C, 4)
    return out_q, out_s, x, t[:, :, S:o_S, :]

def per_channel_dequant(x, q, s):
    B, H, S, C = q.shape
    laq_cuda.dequant(x, q, s, B, H, S, C, 4)
    return x