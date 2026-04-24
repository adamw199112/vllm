import torch
import triton
import triton.language as tl

torch.manual_seed(42)
B, T, H, V = 1, 16, 4, 32

v = torch.randn(B, T, H, V, dtype=torch.float32).cuda()

@triton.jit
def debug_load_kernel(
    v,
    out,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    MAX_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    t_idx = tl.arange(0, BT)
    v_idx = tl.arange(0, V)
    
    v_offsets = ((t_idx)[:, None] * H + i_h) * V + v_idx[None, :]
    v_mask = (t_idx[:, None] < MAX_T) & (v_idx[None, :] < V)
    
    b_v = tl.load(v + v_offsets, mask=v_mask, other=0.0)
    out_offsets = i_bh * BT * V + t_idx[:, None] * V + v_idx[None, :]
    out_mask = t_idx[:, None] < MAX_T
    tl.store(out + out_offsets, b_v, mask=out_mask)

out = torch.zeros(B * H, T, V, dtype=torch.float32).cuda()
debug_load_kernel[(1, B*H)](v, out, H, V, BT=T, MAX_T=T)

# Compare
print('v shape:', v.shape)
print('out shape:', out.shape)
print('v[0,5,1,:]:', v[0,5,1,:].cpu()[:8])
print('out[1,5,:]:', out[1,5,:].cpu()[:8])

# Check specific indices
for t in range(5):
    for h in range(H):
        for vi in range(min(V, 4)):
            diff = (v[0,t,h,vi] - out[h,t,vi]).abs()
            if diff > 1e-5:
                print(f'MISMATCH at v[0,{t},{h},{vi}]: v={v[0,t,h,vi].item():.4f}, out={out[h,t,vi].item():.4f}, diff={diff.item():.6f}')
print('Done checking')