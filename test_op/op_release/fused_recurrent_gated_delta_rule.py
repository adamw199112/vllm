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
    for i_nh in range(N * HV):
        i_n = i_nh // HV
        i_hv = i_nh % HV
        i_h = i_hv // (HV // H)

        for i_v in range(NV):
            for i_k in range(NK):
                if IS_VARLEN:
                    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
                    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
                    all_T = T
                    T_val = eos - bos
                else:
                    bos = i_n * T
                    eos = i_n * T + T
                    all_T = B * T
                    T_val = T

                if T_val > 0:
                    o_k = i_k * BK + tl.arange(0, BK)
                    o_v = i_v * BV + tl.arange(0, BV)

                    p_q = q + (bos * H + i_h) * K + o_k
                    p_k = k + (bos * H + i_h) * K + o_k
                    p_v = v + (bos * HV + i_hv) * V + o_v
                    if IS_BETA_HEADWISE:
                        p_beta = beta + (bos * HV + i_hv) * V + o_v
                    else:
                        p_beta = beta + bos * HV + i_hv

                    if not IS_KDA:
                        p_g = g + bos * HV + i_hv
                    else:
                        p_gk = g + (bos * HV + i_hv) * K + o_k

                    p_o = o + ((i_k * all_T + bos) * HV + i_hv) * V + o_v

                    mask_k = o_k < K
                    mask_v = o_v < V
                    mask_h = mask_v[:, None] & mask_k[None, :]

                    b_h = tl.zeros([BV, BK], dtype=tl.float32)
                    if USE_INITIAL_STATE:
                        if IS_CONTINUOUS_BATCHING:
                            if IS_SPEC_DECODING:
                                i_t_init = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                            else:
                                i_t_init = 0
                            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t_init).to(
                                tl.int64
                            )
                            if state_idx > 0:
                                p_h0 = h0 + state_idx * stride_init_state_token
                                p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
                        else:
                            p_h0 = h0 + i_n * HV * V * K
                            p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

                    for i_t in range(0, T_val):
                        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

                        if USE_QK_L2NORM_IN_KERNEL:
                            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                        b_q = b_q * scale

                        if not IS_KDA:
                            b_g = tl.load(p_g).to(tl.float32)
                            b_h *= tl.exp(b_g)
                        else:
                            b_gk = tl.load(p_gk).to(tl.float32)
                            b_h *= tl.exp(b_gk[None, :])

                        b_v -= tl.sum(b_h * b_k[None, :], 1)
                        if IS_BETA_HEADWISE:
                            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
                        else:
                            b_beta = tl.load(p_beta).to(tl.float32)
                        b_v *= b_beta

                        b_h += b_v[:, None] * b_k[None, :]

                        b_o = tl.sum(b_h * b_q[None, :], 1)
                        tl.store(p_o, b_o.to(tl.float16), mask=mask_v)

                        if INPLACE_FINAL_STATE:
                            final_state_idx = tl.load(
                                ssm_state_indices + i_n * stride_indices_seq + i_t
                            ).to(tl.int64)
                            if final_state_idx > 0:
                                p_ht = ht + final_state_idx * stride_final_state_token
                                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                tl.store(p_ht, b_h.to(tl.float16), mask=mask_h)
                        else:
                            p_ht = ht + (bos + i_t) * stride_final_state_token
                            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                            tl.store(p_ht, b_h.to(tl.float16), mask=mask_h)

                        p_q += H * K
                        p_k += H * K
                        p_o += HV * V
                        p_v += HV * V
                        if not IS_KDA:
                            p_g += HV
                        else:
                            p_gk += HV * K
                        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


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
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = min(triton.next_power_of_2(K), 32)
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
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
    if NK == 1:
        o = o.squeeze(0)
    else:
        o = o.sum(dim=0)
    return o, final_state
