import torch
import sys
sys.path.insert(0, '/home/adam.wang/work/vllm')

torch.manual_seed(42)
B, T, H, V, BT, BV = 1, 16, 4, 32, 16, 32

v = torch.randn(B, T, H, V, dtype=torch.float32).cuda()
A = torch.randn(B, T, H, BT, dtype=torch.float32).cuda()

import triton
import triton.language as tl

@triton.jit
def test_block_ptr(v, out, H: tl.constexpr, V: tl.constexpr, T: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_bh = tl.program_id(0)
    i_b = 0
    i_h = i_bh % H
    
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (i_b * T + i_h) * V,
            (T, V),
            (H * V, 1),
            (0, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        off = i_bh * BT * V + tl.arange(0, BT)[:, None] * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(out + off, b_v)

@triton.jit
def test_offset(v, out, H: tl.constexpr, V: tl.constexpr, T: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_bh = tl.program_id(0)
    i_b = 0
    i_h = i_bh % H
    
    for i_v in range(tl.cdiv(V, BV)):
        t_idx = tl.arange(0, BT)
        v_idx = i_v * BV + tl.arange(0, BV)
        v_offsets = (t_idx[:, None] * H + i_h) * V + v_idx[None, :]
        v_mask = t_idx[:, None] < T
        b_v = tl.load(v + v_offsets, mask=v_mask, other=0.0)
        off = i_bh * BT * V + t_idx[:, None] * V + v_idx[None, :]
        tl.store(out + off, b_v)

out1 = torch.zeros(B * H, T, V, dtype=torch.float32).cuda()
test_block_ptr[(B*H,)](v, out1, H, V, T, BT, BV)

out2 = torch.zeros(B * H, T, V, dtype=torch.float32).cuda()
test_offset[(B*H,)](v, out2, H, V, T, BT, BV)

print('Match:', torch.allclose(out1, out2))
print('Max diff:', (out1 - out2).abs().max().item())
print('Block ptr out[0,4,:8]:')
print(out1[0, 4, :8])
print('Offset out[0,4,:8]:')
print(out2[0, 4, :8])