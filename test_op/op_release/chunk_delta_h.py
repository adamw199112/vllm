import torch
import triton
import triton.language as tl


def cdiv(a, b):
    return (a + b - 1) // b


def prepare_chunk_indices(cu_seqlens, chunk_size):
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = cdiv(seq_lens, chunk_size)
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens, chunk_size):
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = cdiv(seq_lens, chunk_size)
    return torch.cat([cu_seqlens.new_tensor([0]), num_chunks]).cumsum(-1)


@triton.jit
def chunk_gated_delta_rule_fwd_kernel(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    N: tl.constexpr,
    NV: tl.constexpr,
    NT: tl.constexpr,
):
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    for i_nh in range(N * H):
        i_n = i_nh // H
        i_h = i_nh % H
        k_head = i_h // (H // Hg)

        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_cur = eos - bos
            local_NT = tl.cdiv(T_cur, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        else:
            bos = i_n * T
            eos = i_n * T + T
            T_cur = T
            local_NT = NT
            boh = i_n * NT

        h_p = h + ((boh * H + i_h) * V * K).to(tl.int64)
        v_p = v + ((bos * H + i_h) * V).to(tl.int64)
        k_p = k + ((bos * Hg + k_head) * K).to(tl.int64)
        w_p = w + ((bos * H + i_h) * K).to(tl.int64)
        if SAVE_NEW_VALUE:
            v_new_p = v_new + ((bos * H + i_h) * V).to(tl.int64)
        if USE_INITIAL_STATE:
            h0_p = h0 + i_nh * V * K
        if STORE_FINAL_STATE:
            ht_p = ht + i_nh * V * K

        for i_v in range(NV):
            b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
            if K > 64:
                b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
            if K > 128:
                b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
            if K > 192:
                b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

            if USE_INITIAL_STATE:
                v_offsets_h0 = i_v * BV + tl.arange(0, BV)
                k_offsets_h0 = tl.arange(0, 64)
                h0_offsets = (v_offsets_h0[:, None] * K + k_offsets_h0[None, :]).to(tl.int64)
                h0_mask = (v_offsets_h0[:, None] < V) & (k_offsets_h0[None, :] < K)
                b_h1 += tl.load(h0_p + h0_offsets, mask=h0_mask, other=0.0).to(tl.float16)
                if K > 64:
                    k_offsets_h0_2 = 64 + tl.arange(0, 64)
                    h0_offsets_2 = (v_offsets_h0[:, None] * K + k_offsets_h0_2[None, :]).to(tl.int64)
                    h0_mask_2 = (v_offsets_h0[:, None] < V) & (k_offsets_h0_2[None, :] < K)
                    b_h2 += tl.load(h0_p + h0_offsets_2, mask=h0_mask_2, other=0.0).to(tl.float16)
                if K > 128:
                    k_offsets_h0_3 = 128 + tl.arange(0, 64)
                    h0_offsets_3 = (v_offsets_h0[:, None] * K + k_offsets_h0_3[None, :]).to(tl.int64)
                    h0_mask_3 = (v_offsets_h0[:, None] < V) & (k_offsets_h0_3[None, :] < K)
                    b_h3 += tl.load(h0_p + h0_offsets_3, mask=h0_mask_3, other=0.0).to(tl.float16)
                if K > 192:
                    k_offsets_h0_4 = 192 + tl.arange(0, 64)
                    h0_offsets_4 = (v_offsets_h0[:, None] * K + k_offsets_h0_4[None, :]).to(tl.int64)
                    h0_mask_4 = (v_offsets_h0[:, None] < V) & (k_offsets_h0_4[None, :] < K)
                    b_h4 += tl.load(h0_p + h0_offsets_4, mask=h0_mask_4, other=0.0).to(tl.float16)

            for i_t in range(local_NT):
                v_offsets_h = i_v * BV + tl.arange(0, BV)
                k_offsets = tl.arange(0, 64)
                h_load_offsets = (v_offsets_h[:, None] * K + k_offsets[None, :]).to(tl.int64)
                h_load_mask = (v_offsets_h[:, None] < V) & (k_offsets[None, :] < K)
                b_h1 = tl.load(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets, mask=h_load_mask, other=0.0)
                tl.store(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets, b_h1, mask=h_load_mask)
                if K > 64:
                    k_offsets_2 = 64 + tl.arange(0, 64)
                    h_load_offsets_2 = (v_offsets_h[:, None] * K + k_offsets_2[None, :]).to(tl.int64)
                    h_load_mask_2 = (v_offsets_h[:, None] < V) & (k_offsets_2[None, :] < K)
                    b_h2 = tl.load(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets_2, mask=h_load_mask_2, other=0.0)
                    tl.store(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets_2, b_h2, mask=h_load_mask_2)
                if K > 128:
                    k_offsets_3 = 128 + tl.arange(0, 64)
                    h_load_offsets_3 = (v_offsets_h[:, None] * K + k_offsets_3[None, :]).to(tl.int64)
                    h_load_mask_3 = (v_offsets_h[:, None] < V) & (k_offsets_3[None, :] < K)
                    b_h3 = tl.load(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets_3, mask=h_load_mask_3, other=0.0)
                    tl.store(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets_3, b_h3, mask=h_load_mask_3)
                if K > 192:
                    k_offsets_4 = 192 + tl.arange(0, 64)
                    h_load_offsets_4 = (v_offsets_h[:, None] * K + k_offsets_4[None, :]).to(tl.int64)
                    h_load_mask_4 = (v_offsets_h[:, None] < V) & (k_offsets_4[None, :] < K)
                    b_h4 = tl.load(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets_4, mask=h_load_mask_4, other=0.0)
                    tl.store(h_p + i_t.to(tl.int64) * stride_h + h_load_offsets_4, b_h4, mask=h_load_mask_4)

                p_w = tl.make_block_ptr(w_p, (T_cur, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
                if K > 64:
                    p_w = tl.make_block_ptr(w_p, (T_cur, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
                if K > 128:
                    p_w = tl.make_block_ptr(w_p, (T_cur, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
                if K > 192:
                    p_w = tl.make_block_ptr(w_p, (T_cur, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))

                t_offsets_v = i_t * BT + tl.arange(0, BT)
                v_offsets_load = t_offsets_v[:, None] * stride_v + (i_v * BV + tl.arange(0, BV))[None, :]
                v_mask = (t_offsets_v[:, None] < T_cur) & ((i_v * BV + tl.arange(0, BV))[None, :] < V)
                b_v_data = tl.load(v_p + v_offsets_load, mask=v_mask, other=0.0)
                b_v = b_v_data - b_v

                if SAVE_NEW_VALUE:
                    v_new_offsets = t_offsets_v[:, None] * stride_v + (i_v * BV + tl.arange(0, BV))[None, :]
                    v_new_mask = (t_offsets_v[:, None] < T_cur) & ((i_v * BV + tl.arange(0, BV))[None, :] < V)
                    tl.store(v_new_p + v_new_offsets, b_v, mask=v_new_mask)

                last_idx = min((i_t.to(tl.int64) + 1) * BT, T_cur) - 1

                if USE_G:
                    m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T_cur
                    g_p = g + (bos * H + i_h)
                    b_g_last = tl.load(g_p + last_idx * H)
                    t_offsets_g = i_t * BT + tl.arange(0, BT)
                    b_g = tl.load(g_p + t_offsets_g * H, mask=t_offsets_g < T_cur, other=0.0)
                    b_v = b_v * tl.where(m_t, tl.math.exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = tl.math.exp(b_g_last)
                    b_h1 *= b_g_last
                    if K > 64:
                        b_h2 *= b_g_last
                    if K > 128:
                        b_h3 *= b_g_last
                    if K > 192:
                        b_h4 *= b_g_last

                if USE_GK:
                    gk_p = gk + (bos * H * K + i_h * K)
                    o_k1 = tl.arange(0, 64)
                    b_gk_last1 = tl.load(gk_p + last_idx * H * K + o_k1, mask=(o_k1 < K), other=0.0)
                    b_h1 *= tl.math.exp(b_gk_last1)[None, :]
                    if K > 64:
                        o_k2 = 64 + o_k1
                        b_gk_last2 = tl.load(gk_p + last_idx * H * K + o_k2, mask=(o_k2 < K), other=0.0)
                        b_h2 *= tl.math.exp(b_gk_last2)[None, :]
                    if K > 128:
                        o_k3 = 128 + o_k1
                        b_gk_last3 = tl.load(gk_p + last_idx * H * K + o_k3, mask=(o_k3 < K), other=0.0)
                        b_h3 *= tl.math.exp(b_gk_last3)[None, :]
                    if K > 192:
                        o_k4 = 192 + o_k1
                        b_gk_last4 = tl.load(gk_p + last_idx * H * K + o_k4, mask=(o_k4 < K), other=0.0)
                        b_h4 *= tl.math.exp(b_gk_last4)[None, :]

                b_v = b_v.to(k.dtype.element_ty)

                p_k = tl.make_block_ptr(k_p, (K, T_cur), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h1 += tl.trans(tl.dot(b_k, b_v))
                if K > 64:
                    p_k = tl.make_block_ptr(k_p, (K, T_cur), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h2 += tl.trans(tl.dot(b_k, b_v))
                if K > 128:
                    p_k = tl.make_block_ptr(k_p, (K, T_cur), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h3 += tl.trans(tl.dot(b_k, b_v))
                if K > 192:
                    p_k = tl.make_block_ptr(k_p, (K, T_cur), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h4 += tl.trans(tl.dot(b_k, b_v))

            if STORE_FINAL_STATE:
                ht_offsets = (i_v * BV + tl.arange(0, BV))[:, None] * K + tl.arange(0, 64)[None, :]
                ht_mask = ((i_v * BV + tl.arange(0, BV))[:, None] < V) & (tl.arange(0, 64)[None, :] < K)
                tl.store(ht_p + ht_offsets, b_h1, mask=ht_mask)
                if K > 64:
                    ht_offsets_2 = (i_v * BV + tl.arange(0, BV))[:, None] * K + (64 + tl.arange(0, 64))[None, :]
                    ht_mask_2 = ((i_v * BV + tl.arange(0, BV))[:, None] < V) & (((64 + tl.arange(0, 64)))[None, :] < K)
                    tl.store(ht_p + ht_offsets_2, b_h2, mask=ht_mask_2)
                if K > 128:
                    ht_offsets_3 = (i_v * BV + tl.arange(0, BV))[:, None] * K + (128 + tl.arange(0, 64))[None, :]
                    ht_mask_3 = ((i_v * BV + tl.arange(0, BV))[:, None] < V) & (((128 + tl.arange(0, 64)))[None, :] < K)
                    tl.store(ht_p + ht_offsets_3, b_h3, mask=ht_mask_3)
                if K > 192:
                    ht_offsets_4 = (i_v * BV + tl.arange(0, BV))[:, None] * K + (192 + tl.arange(0, 64))[None, :]
                    ht_mask_4 = ((i_v * BV + tl.arange(0, BV))[:, None] < V) & (((192 + tl.arange(0, 64)))[None, :] < K)
                    tl.store(ht_p + ht_offsets_4, b_h4, mask=ht_mask_4)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    assert K <= 256, "current kernel does not support head dimension larger than 256."

    NV = cdiv(V, BV)
    IS_VARLEN = cu_seqlens is not None

    h = k.new_zeros(B, NT, H, V, K)
    final_state = k.new_zeros(N, H, V, K) if output_final_state else k.new_zeros(1, 1, 1, 1, 1)
    v_new = torch.empty_like(u) if save_new_value else torch.empty_like(u)

    USE_G = g is not None
    USE_GK = gk is not None
    USE_INITIAL_STATE = initial_state is not None
    STORE_FINAL_STATE = output_final_state

    if not USE_G:
        g = k.new_zeros(1, 1, 1)
    if not USE_GK:
        gk = k.new_zeros(1, 1, 1, 1)
    if not USE_INITIAL_STATE:
        initial_state = k.new_zeros(1, 1, 1, 1)
    if not IS_VARLEN:
        cu_seqlens = k.new_zeros(2, dtype=torch.int32)
        chunk_offsets = k.new_zeros(1, dtype=torch.int32)

    chunk_gated_delta_rule_fwd_kernel[1, 1](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_G=USE_G,
        USE_GK=USE_GK,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        STORE_FINAL_STATE=STORE_FINAL_STATE,
        SAVE_NEW_VALUE=save_new_value,
        IS_VARLEN=IS_VARLEN,
        N=N,
        NV=NV,
        NT=NT,
    )

    return h, v_new, final_state if output_final_state else None
