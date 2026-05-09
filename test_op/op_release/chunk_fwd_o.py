import torch
import triton
import triton.language as tl


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute chunk indices for variable length sequences."""
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
    cu_seqlens,
    chunk_indices,
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
    IS_VARLEN: tl.constexpr,
    B: tl.constexpr,
    NT: tl.constexpr,
    NV: tl.constexpr,
):
    for i_bh in range(B * H):
        i_b = i_bh // H
        i_h = i_bh % H

        for i_t in range(NT):
            if IS_VARLEN:
                i_tg = i_t
                i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
                i_t_chunk = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
                bos = tl.load(cu_seqlens + i_n).to(tl.int32)
                eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
                T_seq = eos - bos
            else:
                i_tg = i_b * NT + i_t
                i_t_chunk = i_t
                bos = i_b * T
                eos = i_b * T + T
                T_seq = T

            q_base = q + (bos * Hg + i_h // (H // Hg)) * K
            k_base = k + (bos * Hg + i_h // (H // Hg)) * K
            v_base = v + (bos * H + i_h) * V
            o_base = o + (bos * H + i_h) * V
            h_base = h + (i_tg * H + i_h).to(tl.int64) * V * K

            if USE_G:
                g_base = g + bos * H + i_h

            for i_v in range(NV):
                b_o_acc = tl.zeros([BT, BV], dtype=tl.float32)
                b_A = tl.zeros([BT, BT], dtype=tl.float32)

                for i_k in range(tl.cdiv(K, BK)):
                    p_q = tl.make_block_ptr(
                        q_base, (T_seq, K), (Hg * K, 1),
                        (i_t_chunk * BT, i_k * BK), (BT, BK), (1, 0)
                    )
                    p_k = tl.make_block_ptr(
                        k_base, (K, T_seq), (1, Hg * K),
                        (i_k * BK, i_t_chunk * BT), (BK, BT), (0, 1)
                    )

                    t_offsets_h = i_v * BV + tl.arange(0, BV)
                    k_offsets = i_k * BK + tl.arange(0, BK)
                    h_offsets = t_offsets_h[:, None] * K + k_offsets[None, :]
                    h_mask = (t_offsets_h[:, None] < V) & (k_offsets[None, :] < K)

                    b_h = tl.load(h_base + h_offsets, mask=h_mask, other=0.0)
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))

                    b_o_acc += tl.dot(b_q.to(tl.float16), tl.trans(b_h).to(tl.float16))
                    b_A += tl.dot(b_q, b_k)

                if USE_G:
                    offs_t = i_t_chunk * BT + tl.arange(0, BT)
                    ptrs = g_base + offs_t * H
                    b_g = tl.load(ptrs, mask=offs_t < T_seq, other=0)
                    b_o_acc = b_o_acc * tl.exp(b_g.to(tl.float32))[:, None]
                    b_A = b_A * tl.exp((b_g[:, None] - b_g[None, :]).to(tl.float32))

                o_t = i_t_chunk * BT + tl.arange(0, BT)
                m_t = o_t < T_seq
                m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
                b_A = tl.where(m_A, b_A, 0)

                t_offsets_o = i_t_chunk * BT + tl.arange(0, BT)
                v_idx_offsets = i_v * BV + tl.arange(0, BV)
                o_offsets_full = t_offsets_o[:, None] * H * V + v_idx_offsets[None, :]
                o_mask = (t_offsets_o[:, None] < T_seq) & (v_idx_offsets[None, :] < V)

                p_v = tl.make_block_ptr(
                    v_base, (T_seq, V), (H * V, 1),
                    (i_t_chunk * BT, i_v * BV), (BT, BV), (1, 0)
                )
                b_v = tl.load(p_v, boundary_check=(0, 1))

                b_o_acc = b_o_acc * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
                tl.store(o_base + o_offsets_full, b_o_acc.to(tl.float16), mask=o_mask)


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
    BK: int = 64,
    BV: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    if cu_seqlens is None:
        NT = triton.cdiv(T, BT)
        IS_VARLEN = False
        cu_seqlens_ptr = torch.zeros(2, dtype=torch.int32, device=q.device)
        chunk_indices_ptr = torch.zeros((NT, 2), dtype=torch.int32, device=q.device)
    else:
        NT = len(chunk_indices)
        IS_VARLEN = True
        cu_seqlens_ptr = cu_seqlens
        chunk_indices_ptr = chunk_indices

    USE_G = g is not None
    if g is None:
        g_ptr = torch.zeros(1, dtype=q.dtype, device=q.device)
    else:
        g_ptr = g

    if scale is None:
        scale = K ** -0.5

    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)

    chunk_fwd_kernel_o[1, 1](
        q,
        k,
        v,
        h,
        g_ptr,
        o,
        cu_seqlens_ptr,
        chunk_indices_ptr,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=USE_G,
        IS_VARLEN=IS_VARLEN,
        B=B,
        NT=NT,
        NV=NV,
    )
    return o
