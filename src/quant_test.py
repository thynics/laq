# quant_test.py
import torch
import os
from quant import per_channel_dequant, per_channel_quant        # <— name 对应 setup.py 里的 name='laq'

def test_quant():
    B, H, S, C = 2, 3, 128, 16
    bit_width = 8
    x = torch.randn(B, H, S, C, dtype=torch.float32, device='cuda')
    x = x * 10
    print(x.view(-1)[:10])
    print("----------------")
    q, s, left, right = per_channel_quant(x)
    print(q.view(-1)[:10])
    print(s[0])
    per_channel_dequant(left, q, s)


if __name__=="__main__":
    assert torch.cuda.is_available()
    test_quant()
