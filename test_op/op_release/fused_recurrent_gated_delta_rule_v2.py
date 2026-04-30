import torch
import triton
import triton.language as tl


@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    NK: tl.constexpr,
    BK_FULL: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    # Implementation notes
    # --------------------
    # The kernel supports arbitrary NK (>=1) with a fixed BK (e.g. BK=32). The
    # hidden-state tile `b_h` lives over the padded K dimension, with column
    # width `BK_FULL = NK * BK`. q / k / final-state memory accesses are issued
    # as BK-sized chunks via `tl.static_range(NK)` loops; partial reductions
    # for the per-timestep output `b_o` and delta-rule term `b_delta` are kept
    # in BV-wide register vectors and only written back once per timestep.
    for i_nh in range(N * HV):
        i_n = i_nh // HV
        i_hv = i_nh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_val = eos - bos
        else:
            bos = i_n * T
            T_val = T

        if T_val > 0:
            for i_v in range(NV):
                o_v = i_v * BV + tl.arange(0, BV)
                mask_v = o_v < V

                # Column indices / mask for the full padded K dimension.
                o_k_full = tl.arange(0, BK_FULL)
                mask_k_full = o_k_full < K
                mask_h_full = mask_v[:, None] & mask_k_full[None, :]

                # Hidden state tile (single tile, full padded K).
                b_h = tl.zeros([BV, BK_FULL], dtype=tl.float32)

                if USE_INITIAL_STATE:
                    if IS_CONTINUOUS_BATCHING:
                        if IS_SPEC_DECODING:
                            i_t_init = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                        else:
                            i_t_init = 0
                        state_idx = tl.load(
                            ssm_state_indices + i_n * stride_indices_seq + i_t_init
                        ).to(tl.int64)
                        if state_idx > 0:
                            p_h0_base = h0 + state_idx * stride_init_state_token
                    else:
                        p_h0_base = h0 + i_n * HV * V * K
                    # Load initial state in BK-sized column chunks.
                    for i_k in tl.static_range(NK):
                        in_block = (o_k_full >= i_k * BK) & (o_k_full < (i_k + 1) * BK)
                        p_h0 = (
                            p_h0_base
                            + i_hv * V * K
                            + o_v[:, None] * K
                            + o_k_full[None, :]
                        )
                        b_h += tl.load(
                            p_h0,
                            mask=mask_h_full & in_block[None, :],
                            other=0,
                        ).to(tl.float32)

                for i_t in range(0, T_val):
                    # V-side loads (independent of K).
                    b_v = tl.load(
                        v + ((bos + i_t) * HV + i_hv) * V + o_v,
                        mask=mask_v, other=0,
                    ).to(tl.float32)
                    if IS_BETA_HEADWISE:
                        b_beta = tl.load(
                            beta + ((bos + i_t) * HV + i_hv) * V + o_v,
                            mask=mask_v, other=0,
                        ).to(tl.float32)
                    else:
                        b_beta = tl.load(beta + (bos + i_t) * HV + i_hv).to(tl.float32)
                    if not IS_KDA:
                        b_g = tl.load(g + (bos + i_t) * HV + i_hv).to(tl.float32)

                    # L2-norm pre-pass — sum of squares accumulated in scalars
                    # via BK-chunked reads.
                    if USE_QK_L2NORM_IN_KERNEL:
                        q_sq_sum = tl.zeros([], dtype=tl.float32)
                        k_sq_sum = tl.zeros([], dtype=tl.float32)
                        for i_k in tl.static_range(NK):
                            o_k = i_k * BK + tl.arange(0, BK)
                            mask_k = o_k < K
                            b_q_chunk = tl.load(
                                q + ((bos + i_t) * H + i_h) * K + o_k,
                                mask=mask_k, other=0,
                            ).to(tl.float32)
                            b_k_chunk = tl.load(
                                k + ((bos + i_t) * H + i_h) * K + o_k,
                                mask=mask_k, other=0,
                            ).to(tl.float32)
                            q_sq_sum += tl.sum(b_q_chunk * b_q_chunk)
                            k_sq_sum += tl.sum(b_k_chunk * b_k_chunk)
                        q_norm = tl.sqrt(q_sq_sum + 1e-6)
                        k_norm = tl.sqrt(k_sq_sum + 1e-6)

                    # Decay b_h.
                    if not IS_KDA:
                        b_h *= tl.exp(b_g)
                    else:
                        b_gk_full = tl.load(
                            g + ((bos + i_t) * HV + i_hv) * K + o_k_full,
                            mask=mask_k_full, other=0,
                        ).to(tl.float32)
                        b_h *= tl.exp(b_gk_full[None, :])

                    # Load b_q / b_k as full BK_FULL vectors using BK-chunked
                    # reads; intermediate `b_q_full` / `b_k_full` are register
                    # vectors padded with zeros outside the active chunk.
                    b_q_full = tl.zeros([BK_FULL], dtype=tl.float32)
                    b_k_full = tl.zeros([BK_FULL], dtype=tl.float32)
                    for i_k in tl.static_range(NK):
                        in_block = (o_k_full >= i_k * BK) & (o_k_full < (i_k + 1) * BK)
                        b_q_full += tl.load(
                            q + ((bos + i_t) * H + i_h) * K + o_k_full,
                            mask=mask_k_full & in_block, other=0,
                        ).to(tl.float32)
                        b_k_full += tl.load(
                            k + ((bos + i_t) * H + i_h) * K + o_k_full,
                            mask=mask_k_full & in_block, other=0,
                        ).to(tl.float32)
                    if USE_QK_L2NORM_IN_KERNEL:
                        b_q_full = b_q_full / q_norm
                        b_k_full = b_k_full / k_norm
                    b_q_full = b_q_full * scale

                    # delta-rule: register accumulator (length BV).
                    b_delta = tl.sum(b_h * b_k_full[None, :], 1)
                    b_v_new = (b_v - b_delta) * b_beta
                    b_h += b_v_new[:, None] * b_k_full[None, :]

                    # Output: register accumulator, store once per timestep.
                    b_o = tl.sum(b_h * b_q_full[None, :], 1)
                    p_o = o + ((bos + i_t) * HV + i_hv) * V + o_v
                    tl.store(p_o, b_o.to(tl.float16), mask=mask_v)

                    # Store final state in BK-sized column chunks.
                    if INPLACE_FINAL_STATE:
                        final_state_idx = tl.load(
                            ssm_state_indices + i_n * stride_indices_seq + i_t
                        ).to(tl.int64)
                        if final_state_idx > 0:
                            p_ht_base = (
                                ht
                                + final_state_idx * stride_final_state_token
                                + i_hv * V * K
                            )
                            for i_k in tl.static_range(NK):
                                in_block = (o_k_full >= i_k * BK) & (o_k_full < (i_k + 1) * BK)
                                p_ht = p_ht_base + o_v[:, None] * K + o_k_full[None, :]
                                tl.store(
                                    p_ht,
                                    b_h.to(tl.float16),
                                    mask=mask_h_full & in_block[None, :],
                                )
                    else:
                        p_ht_base = (
                            ht
                            + (bos + i_t) * stride_final_state_token
                            + i_hv * V * K
                        )
                        for i_k in tl.static_range(NK):
                            in_block = (o_k_full >= i_k * BK) & (o_k_full < (i_k + 1) * BK)
                            p_ht = p_ht_base + o_v[:, None] * K + o_k_full[None, :]
                            tl.store(
                                p_ht,
                                b_h.to(tl.float16),
                                mask=mask_h_full & in_block[None, :],
                            )


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
   implementation of the fused recurrent gated delta rule.

    Computes per-step:
        h *= exp(g)           # decay
        v_new = v - h @ k     # delta rule
        v_new *= beta          # beta gate
        h += v_new ⊗ k         # update
        o = h @ q              # output

    Args:
        q: [B, T, H, K]  queries
        k: [B, T, H, K]  keys
        v: [B, T, HV, V]  values (GVA: HV >= H)
        g: [B, T, HV]  decays
        beta: [B, T, HV] or [B, T, HV, V]  betas
        scale: scale factor for q
        initial_state: [N, HV, V, K]  initial hidden states
        inplace_final_state: if True, final_state shares storage with initial_state
        cu_seqlens: [N+1]  cumulative sequence lengths (varlen)
        ssm_state_indices: indices for state mapping (ignored here)
        num_accepted_tokens: for spec decoding (ignored here)
        use_qk_l2norm_in_kernel: whether to L2-normalize q and k

    Returns:
        o: [B, T, HV, V]  output
        final_state: per-sequence final hidden states [N, HV, V, K]
    """
    
    
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = 32
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    BK_FULL = NK * BK
    assert BK_FULL >= K, "BK_FULL must cover K"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(*v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    is_varlen = cu_seqlens is not None
    is_continuous_batching = ssm_state_indices is not None
    is_spec_decoding = num_accepted_tokens is not None
    use_initial_state = initial_state is not None
    is_beta_headwise = beta.ndim == v.ndim

    cu_seqlens_ptr = cu_seqlens if cu_seqlens is not None else torch.zeros(2, dtype=torch.int32, device=q.device)
    ssm_state_indices_ptr = ssm_state_indices if ssm_state_indices is not None else torch.zeros(1, dtype=torch.int32, device=q.device)
    num_accepted_tokens_ptr = num_accepted_tokens if num_accepted_tokens is not None else torch.zeros(1, dtype=torch.int32, device=q.device)

    fused_recurrent_gated_delta_rule_fwd_kernel[1, 1](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens_ptr,
        ssm_state_indices=ssm_state_indices_ptr,
        num_accepted_tokens=num_accepted_tokens_ptr,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NV=NV,
        NK=NK,
        BK_FULL=BK_FULL,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        USE_INITIAL_STATE=use_initial_state,
        IS_BETA_HEADWISE=is_beta_headwise,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_VARLEN=is_varlen,
        IS_CONTINUOUS_BATCHING=is_continuous_batching,
        IS_SPEC_DECODING=is_spec_decoding,
        IS_KDA=False,
    )
    return o, final_state
