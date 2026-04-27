import torch
import triton
import triton.language as tl


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@triton.jit
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    B: tl.constexpr,
    NT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    i_t = tl.program_id(1)
    i_tg = i_b * NT + i_t
    bos = i_b * T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float16)
    b_A = tl.zeros([BT, BT], dtype=tl.float16)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))

        t_offsets_h = tl.arange(0, BV)[:, None]
        k_offsets_h = i_k * BK + tl.arange(0, BK)[None, :]
        h_offsets = t_offsets_h * K + k_offsets_h
        h_mask = (t_offsets_h < V) & (k_offsets_h < K)
        b_h = tl.load(h + h_offsets, mask=h_mask, other=0.0).to(tl.float16)

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float16)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float16)

        b_o += tl.dot(b_q, tl.trans(b_h)).to(tl.float16)
        b_A += tl.dot(b_q, b_k).to(tl.float16)

    if USE_G:
        g += bos * H + i_h
        offs_t = i_t * BT + tl.arange(0, BT)
        ptrs = g + offs_t * H
        b_g = tl.load(ptrs, mask=offs_t < T, other=0).to(tl.float32)
        b_o = b_o * (tl.exp(b_g)[:, None]).to(tl.float16)
        b_A = b_A * (tl.exp(b_g[:, None] - b_g[None, :])).to(tl.float16)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0.0)

    v_offsets = i_t * BT + tl.arange(0, BT)
    v_idx = tl.arange(0, BV)
    o_offsets = v_offsets[:, None] * H * V + v_idx[None, :]
    o_mask = (v_offsets[:, None] < T) & (v_idx[None, :] < V)

    p_v = tl.make_block_ptr(v, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float16)

    b_o = b_o * scale + tl.dot(b_A, b_v) * scale
    tl.store(o + o_offsets, b_o.to(tl.float16), mask=o_mask)


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    if cu_seqlens is None:
        NT = triton.cdiv(T, BT)
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)

    if scale is None:
        scale = K ** -0.5

    use_g = g is not None

    o = torch.empty_like(v)

    def grid(meta):
        return (B * H, NT)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g if g is not None else q.new_zeros(1),
        o,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=32,
        BV=32,
        USE_G=use_g,
        B=B,
        NT=NT,
    )
    return o