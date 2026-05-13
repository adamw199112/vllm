"""
Fused Chunk Gated Delta Rule - single-program reference kernel (grid=(1,1)).

All loops (chunks, heads) are inside the single Triton program.

Usage:
    conda activate kernel
    python test_op/test_fused_chunk_gated_delta_rule_1x1.py
"""

import torch
import math
import sys

sys.path.insert(0, "/home/adam.wang/work/vllm")

from vllm.triton_utils import tl, triton

from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)


###############################################################################
# CPU reference implementation
###############################################################################
def _exp(x):
    return torch.exp(x)


def chunk_gated_delta_rule_fwd_cpu(
    q, k, v, g, beta, scale, initial_state, output_final_state,
    cu_seqlens, chunk_indices, chunk_offsets, chunk_size=64,
):
    B, T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = chunk_size
    num_chunks = len(chunk_indices)
    N = len(cu_seqlens) - 1

    g_cumsum = torch.empty_like(g, dtype=g.dtype)
    A = torch.zeros(B, T, H, BT, dtype=torch.float32)
    Ai = torch.zeros_like(A)
    w = torch.empty(B, T, H, K, dtype=k.dtype)
    u = torch.empty(B, T, H, V, dtype=k.dtype)
    h = torch.empty(B, num_chunks, H, V, K, dtype=torch.float32)
    v_new = torch.empty(B, T, H, V, dtype=k.dtype)
    final_state = torch.empty(N, H, V, K, dtype=torch.float32)
    o = torch.empty(B, T, H, V, dtype=v.dtype)

    for idx in range(num_chunks):
        i_n = int(chunk_indices[idx, 0].item())
        i_t = int(chunk_indices[idx, 1].item())
        bos = int(cu_seqlens[i_n].item())
        eos = int(cu_seqlens[i_n + 1].item())
        T_seq = eos - bos
        t_start = i_t * BT
        t_end = min(t_start + BT, T_seq)
        cur_BT = t_end - t_start
        if cur_BT <= 0:
            continue

        for h_idx in range(H):
            k_head_idx = h_idx // (H // Hg) if H != Hg else h_idx

            beta_slice = beta[0, bos + t_start : bos + t_end, h_idx]
            k_slice = k[0, bos + t_start : bos + t_end, k_head_idx, :]
            k_beta = k_slice * beta_slice.unsqueeze(1)
            A_chunk = torch.matmul(k_beta, k_slice.T)

            chunk = g[0, bos + t_start : bos + t_end, h_idx].to(torch.float32)
            g_slice = torch.cumsum(chunk, dim=0)
            g_cumsum[0, bos + t_start : bos + t_end, h_idx] = g_slice.to(g.dtype)
            g_diff = g_slice.unsqueeze(1) - g_slice.unsqueeze(0)
            A_chunk = A_chunk * _exp(g_diff)
            causal_mask = torch.tril(torch.ones(cur_BT, cur_BT), diagonal=-1)
            A_chunk = A_chunk * causal_mask
            A[0, bos + t_start : bos + t_end, h_idx, :cur_BT] = A_chunk

            M = torch.eye(cur_BT) + torch.tril(A_chunk)
            X = torch.zeros(cur_BT, cur_BT)
            for j in range(cur_BT):
                X[j, j] = 1.0
                for i in range(j + 1, cur_BT):
                    X[i, j] = -torch.dot(M[i, :i], X[:i, j]) / M[i, i]
            Ai[0, bos + t_start : bos + t_end, h_idx, :cur_BT] = X

            A_slice = X
            g_exp = _exp(g_slice)
            for i_v in range(math.ceil(V / 64)):
                v_s, v_e = i_v * 64, min(i_v * 64 + 64, V)
                v_sliced = v[0, bos + t_start : bos + t_end, h_idx, v_s:v_e]
                u[0, bos + t_start : bos + t_end, h_idx, v_s:v_e] = \
                    torch.matmul(A_slice, v_sliced * beta_slice[:, None])
            for i_k in range(math.ceil(K / 64)):
                k_s, k_e = i_k * 64, min(i_k * 64 + 64, K)
                k_sliced = k[0, bos + t_start : bos + t_end, k_head_idx, k_s:k_e]
                w[0, bos + t_start : bos + t_end, h_idx, k_s:k_e] = \
                    torch.matmul(A_slice, k_sliced * beta_slice[:, None] * g_exp[:, None])

            h_start = initial_state[i_n, h_idx].clone() \
                if initial_state is not None else torch.zeros(V, K)
            h[0, idx, h_idx] = h_start.clone()

            last_idx_in_chunk = t_end - 1
            g_last = g_cumsum[0, bos + last_idx_in_chunk, h_idx]
            h_state_cur = h_start * _exp(g_last)

            for t in range(cur_BT):
                t_idx = bos + t_start + t
                k_head_idx_t = h_idx // (H // Hg) if H != Hg else h_idx
                v_slice = u[0, t_idx, h_idx, :]
                w_slice = w[0, t_idx, h_idx, :]
                v_new_val = v_slice - torch.mm(w_slice.unsqueeze(0), h_start.T).squeeze(0)
                v_new[0, t_idx, h_idx, :] = v_new_val
                g_curr = g_cumsum[0, t_idx, h_idx]
                v_gated = v_new_val * _exp(g_last - g_curr)
                k_t = k[0, t_idx, k_head_idx_t, :]
                h_state_cur += torch.outer(k_t, v_gated).T

            final_state[i_n, h_idx] = h_state_cur.clone()

            q_chunk = q[0, bos + t_start : bos + t_end, k_head_idx, :]
            k_chunk = k[0, bos + t_start : bos + t_end, k_head_idx, :]
            v_chunk = v_new[0, bos + t_start : bos + t_end, h_idx, :]
            h_state = h[0, idx, h_idx]
            o_chunk = torch.matmul(q_chunk, h_state.T)
            A_qk = torch.matmul(q_chunk, k_chunk.T)
            g_chunk = g_cumsum[0, bos + t_start : bos + t_end, h_idx]
            o_chunk = o_chunk * _exp(g_chunk[:, None])
            A_qk = A_qk * _exp(g_chunk[:, None] - g_chunk[None, :])
            mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=torch.bool))
            A_qk = A_qk.masked_fill(~mask, 0)
            o_chunk = (o_chunk + torch.matmul(A_qk, v_chunk)) * scale
            o[0, bos + t_start : bos + t_end, h_idx, :] = o_chunk

    return o, final_state, g_cumsum, A, w, h, v_new


###############################################################################
# Single Program Triton Kernel (grid = (1,)) -- ALL INLINE, NO INNER DEFS
###############################################################################
@triton.jit
def _fused_1x1_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    o_ptr, ws_gcum, ws_A, ws_Ai,
    ws_w, ws_u, ws_vn, ws_h,
    h0_ptr, final_ptr,
    cu_seqlens_ptr, chunk_indices_ptr,
    T: tl.constexpr, H: tl.constexpr, Hg: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    N: tl.constexpr, num_chunks: tl.constexpr,
    scale: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    OUTPUT_FINAL_STATE: tl.constexpr,
):
    BK: tl.constexpr = 64
    BV: tl.constexpr = 64
    NK: tl.constexpr = tl.cdiv(K, BK)
    NV: tl.constexpr = tl.cdiv(V, BV)

    off_bt = tl.arange(0, BT)
    off_bk = tl.arange(0, BK)
    off_bv = tl.arange(0, BV)
    off_v = tl.arange(0, V)
    off_k = tl.arange(0, K)

    s_qk = Hg * K
    s_v = H * V
    s_g = H
    s_A = H * BT
    s_w = H * K
    s_u = H * V

    for ch in range(num_chunks):
        i_n = tl.load(chunk_indices_ptr + ch * 2).to(tl.int32)
        i_t = tl.load(chunk_indices_ptr + ch * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        T_seq = eos - bos
        t0 = i_t * BT
        t1 = tl.minimum(t0 + BT, T_seq)
        cBT = (t1 - t0).to(tl.int32)
        pos = bos + t0

        for h_ in range(H):
            k_h = h_ // (H // Hg) if H != Hg else h_

            # ===== 1. Load g, beta; compute g_cumsum =====
            # Load g into b_g [BT]
            b_g = tl.zeros([BT], dtype=tl.float32)
            b_beta = tl.zeros([BT], dtype=tl.float32)
            for t in range(BT):
                it = pos + t
                mt = it < eos
                gv = tl.load(g_ptr + it * s_g + h_, mask=mt, other=0.0).to(tl.float32)
                bv = tl.load(beta_ptr + it * s_g + h_, mask=mt, other=0.0).to(tl.float32)
                b_g = tl.where(off_bt == t, gv, b_g)
                b_beta = tl.where(off_bt == t, bv, b_beta)

            b_gcum = tl.cumsum(b_g, axis=0)

            for t in range(BT):
                it = pos + t
                gcv = tl.sum(b_gcum * tl.where(off_bt == t, 1.0, 0.0).to(tl.float32))
                tl.store(ws_gcum + it * s_g + h_, gcv, mask=it < eos)

            # ===== 2. A = sum_K tile tl.dot(k_tile, (k*beta)_tile^T) =====
            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for ik in range(NK):
                ks = ik * BK
                b_k1 = tl.zeros([BT, BK], dtype=tl.float32)
                b_k2 = tl.zeros([BT, BK], dtype=tl.float32)
                for t in range(BT):
                    it = pos + t
                    mt = it < eos
                    kv = tl.load(k_ptr + it * s_qk + k_h * K + ks + off_bk,
                                 mask=mt & (ks + off_bk < K), other=0.0).to(tl.float32)
                    bt_t = tl.sum(b_beta * tl.where(off_bt == t, 1.0, 0.0).to(tl.float32))
                    row = tl.where(off_bt[:, None] == t, kv[None, :],
                                   tl.zeros([BT, BK], dtype=tl.float32))
                    b_k1 = b_k1 + row
                    b_k2 = b_k2 + row * bt_t
                b_A += tl.dot(b_k2, tl.trans(b_k1))

            b_A = b_A * tl.exp(b_gcum[:, None] - b_gcum[None, :])
            b_A = b_A * (off_bt[:, None] > off_bt[None, :]).to(tl.float32)

            for i in range(BT):
                row_i = tl.sum(b_A * tl.where(off_bt[:, None] == i, 1.0, 0.0).to(tl.float32),
                               axis=0)
                tl.store(ws_A + (pos + i) * s_A + h_ * BT + off_bt,
                         row_i, mask=((pos + i) < eos) & (off_bt < cBT))

            # ===== 3. Forward substitution: X = (I+L)^{-1}  =====
            b_X = tl.zeros([BT, BT], dtype=tl.float32)
            for j in range(BT):
                b_Xcol = tl.where(off_bt == j, 1.0, 0.0).to(tl.float32)
                for i in range(j + 1, BT):
                    lrow = tl.load(ws_A + (pos + i) * s_A + h_ * BT + off_bt,
                                   mask=((pos + i) < eos) & (off_bt < cBT), other=0.0)
                    m_ij = (off_bt >= j) & (off_bt < i)
                    s_val = tl.sum(lrow * b_Xcol * m_ij.to(tl.float32))
                    b_Xcol = tl.where(off_bt == i, -s_val, b_Xcol)
                col = tl.where(off_bt[None, :] == j, b_Xcol[:, None],
                               tl.zeros([BT, BT], dtype=tl.float32))
                b_X = b_X + col

            for i in range(BT):
                row_i = tl.sum(b_X * tl.where(off_bt[:, None] == i, 1.0, 0.0).to(tl.float32),
                               axis=0)
                tl.store(ws_Ai + (pos + i) * s_A + h_ * BT + off_bt,
                         row_i, mask=((pos + i) < eos) & (off_bt < cBT))

            # ===== 4a. u = X @ (v * beta)  =====
            b_ge = tl.exp(b_gcum)
            for iv in range(NV):
                vs = iv * BV
                b_vb = tl.zeros([BT, BV], dtype=tl.float32)
                for t in range(BT):
                    it = pos + t
                    mt = it < eos
                    vv = tl.load(v_ptr + it * s_v + h_ * V + vs + off_bv,
                                 mask=mt & (vs + off_bv < V), other=0.0).to(tl.float32)
                    bt_t = tl.sum(b_beta * tl.where(off_bt == t, 1.0, 0.0).to(tl.float32))
                    row = tl.where(off_bt[:, None] == t, vv[None, :] * bt_t,
                                   tl.zeros([BT, BV], dtype=tl.float32))
                    b_vb = b_vb + row
                b_u = tl.dot(b_X.to(tl.float32), b_vb)
                for t in range(BT):
                    it = pos + t
                    mt = it < eos
                    row = tl.sum(b_u * tl.where(off_bt[:, None] == t, 1.0, 0.0).to(tl.float32),
                                 axis=0)
                    tl.store(ws_u + it * s_u + h_ * V + vs + off_bv,
                             row, mask=mt & (vs + off_bv < V))

            # ===== 4b. w = X @ (k * beta * exp(gcum))  =====
            for ik in range(NK):
                ks = ik * BK
                b_kb = tl.zeros([BT, BK], dtype=tl.float32)
                for t in range(BT):
                    it = pos + t
                    mt = it < eos
                    kv = tl.load(k_ptr + it * s_qk + k_h * K + ks + off_bk,
                                 mask=mt & (ks + off_bk < K), other=0.0).to(tl.float32)
                    bt_t = tl.sum(b_beta * tl.where(off_bt == t, 1.0, 0.0).to(tl.float32))
                    ge_t = tl.sum(b_ge * tl.where(off_bt == t, 1.0, 0.0).to(tl.float32))
                    row = tl.where(off_bt[:, None] == t, kv[None, :] * bt_t * ge_t,
                                   tl.zeros([BT, BK], dtype=tl.float32))
                    b_kb = b_kb + row
                b_w = tl.dot(b_X.to(tl.float32), b_kb)
                for t in range(BT):
                    it = pos + t
                    mt = it < eos
                    row = tl.sum(b_w * tl.where(off_bt[:, None] == t, 1.0, 0.0).to(tl.float32),
                                 axis=0)
                    tl.store(ws_w + it * s_w + h_ * K + ks + off_bk,
                             row, mask=mt & (ks + off_bk < K))

            # ===== 5a. Load initial hidden state [V, K] =====
            b_hcur = tl.zeros([V, K], dtype=tl.float32)
            if HAS_INITIAL_STATE:
                b_hcur = tl.load(
                    h0_ptr
                    + ((i_n * H + h_) * V * K).to(tl.int64)
                    + (off_v[:, None] * K + off_k[None, :]).to(tl.int64),
                    mask=(off_v[:, None] < V) & (off_k[None, :] < K),
                    other=0.0,
                ).to(tl.float32)

            # snapshot for v_new computation
            b_hs = b_hcur

            # store h for this chunk
            ch_off = ((ch * H + h_) * V * K).to(tl.int64)
            tl.store(
                ws_h + ch_off + (off_v[:, None] * K + off_k[None, :]).to(tl.int64),
                b_hcur.to(tl.float32),
                mask=(off_v[:, None] < V) & (off_k[None, :] < K),
            )

            # ===== 5b. Recurrence =====
            # Absolute position of the last token in the chunk: bos + t1 - 1
            last_abs = bos + t1 - 1
            last_mask = last_abs < eos
            b_gl = tl.load(ws_gcum + last_abs * s_g + h_,
                           mask=last_mask, other=0.0).to(tl.float32)
            decay_val = tl.exp(b_gl).to(tl.float32)

            # Apply chunk-level decay to running state (matching CPU:
            # h_state_cur = h_start * exp(g_last))
            b_hcur = b_hcur * decay_val

            for t in range(BT):
                it = pos + t
                mt = it < eos

                b_ut = tl.load(ws_u + it * s_u + h_ * V + off_v,
                               mask=mt & (off_v < V), other=0.0).to(tl.float32)
                b_wt = tl.load(ws_w + it * s_w + h_ * K + off_k,
                               mask=mt & (off_k < K), other=0.0).to(tl.float32)

                # v_new[t] = u[t] - h_start @ w[t]
                # b_hs: [V, K], b_wt: [K]
                b_wt2d = tl.reshape(b_wt, [K, 1])
                b_wn2d = tl.dot(b_hs.to(tl.float32), b_wt2d.to(tl.float32))
                b_wn = tl.reshape(b_wn2d, [V])
                b_vn = b_ut - b_wn

                tl.store(ws_vn + it * s_v + h_ * V + off_v,
                         b_vn, mask=mt & (off_v < V))

                b_gct = tl.load(ws_gcum + it * s_g + h_,
                                mask=mt, other=0.0).to(tl.float32)
                gate = tl.exp(b_gl - b_gct).to(tl.float32)
                b_vg = b_vn * gate

                b_kt = tl.load(k_ptr + it * s_qk + k_h * K + off_k,
                               mask=mt & (off_k < K), other=0.0).to(tl.float32)

                # Outer product: b_vg[V] (x) b_kt[K] => [V, K]
                # Use broadcast: row v_ = b_vg[v_] * b_kt
                b_update = tl.zeros([V, K], dtype=tl.float32)
                for v_ in range(V):
                    val_v = tl.sum(b_vg * tl.where(off_v == v_, 1.0, 0.0).to(tl.float32))
                    b_update = b_update + tl.where(
                        off_v[:, None] == v_,
                        val_v * tl.reshape(b_kt, [1, K]),
                        tl.zeros([V, K], dtype=tl.float32),
                    )
                b_hcur = b_hcur + b_update

            # Store final state
            if OUTPUT_FINAL_STATE:
                fs_off = ((i_n * H + h_) * V * K).to(tl.int64)
                tl.store(
                    final_ptr + fs_off
                    + (off_v[:, None] * K + off_k[None, :]).to(tl.int64),
                    b_hcur.to(tl.float32),
                    mask=(off_v[:, None] < V) & (off_k[None, :] < K),
                )

            # ===== 6a. Load q, k (for intra-chunk attention) =====
            b_q = tl.zeros([BT, K], dtype=tl.float32)
            b_kc = tl.zeros([BT, K], dtype=tl.float32)
            for t in range(BT):
                it = pos + t
                mt = it < eos
                qv = tl.load(q_ptr + it * s_qk + k_h * K + off_k,
                             mask=mt & (off_k < K), other=0.0).to(tl.float32)
                kv = tl.load(k_ptr + it * s_qk + k_h * K + off_k,
                             mask=mt & (off_k < K), other=0.0).to(tl.float32)
                row_q = tl.where(off_bt[:, None] == t, qv[None, :],
                                 tl.zeros([BT, K], dtype=tl.float32))
                row_k = tl.where(off_bt[:, None] == t, kv[None, :],
                                 tl.zeros([BT, K], dtype=tl.float32))
                b_q = b_q + row_q
                b_kc = b_kc + row_k

            # ===== 6b. o_chunk = q @ h_start^T  =====
            b_o = tl.dot(b_q.to(tl.float32), tl.trans(b_hs.to(tl.float32)))

            # ===== 6c. A_qk = q @ k^T  =====
            b_Aqk = tl.dot(b_q.to(tl.float32), tl.trans(b_kc.to(tl.float32)))

            # ===== 6d. Apply gating  =====
            b_o = b_o * tl.exp(b_gcum)[:, None]
            b_Aqk = b_Aqk * tl.exp(b_gcum[:, None] - b_gcum[None, :])
            b_Aqk = b_Aqk * (off_bt[:, None] >= off_bt[None, :]).to(tl.float32)

            # ===== 6e. Load v_new for this chunk  =====
            b_vnc = tl.zeros([BT, V], dtype=tl.float32)
            for t in range(BT):
                it = pos + t
                mt = it < eos
                vn = tl.load(ws_vn + it * s_v + h_ * V + off_v,
                             mask=mt & (off_v < V), other=0.0).to(tl.float32)
                row = tl.where(off_bt[:, None] == t, vn[None, :],
                               tl.zeros([BT, V], dtype=tl.float32))
                b_vnc = b_vnc + row

            # ===== 6f. A_qk @ v_new and combine  =====
            b_o2 = tl.dot(b_Aqk.to(tl.float32), b_vnc.to(tl.float32))
            b_o = (b_o + b_o2) * scale

            # ===== 6g. Store output =====
            for t in range(BT):
                it = pos + t
                mt = it < eos
                row = tl.sum(b_o * tl.where(off_bt[:, None] == t, 1.0, 0.0).to(tl.float32),
                             axis=0)
                tl.store(o_ptr + it * s_v + h_ * V + off_v,
                         row.to(tl.float32), mask=mt & (off_v < V))


###############################################################################
# Python wrapper
###############################################################################
def fused_chunk_gated_delta_rule_1x1(
    q, k, v, g, beta, scale, initial_state, output_final_state,
    cu_seqlens, chunk_indices, chunk_offsets, chunk_size=64,
):
    B, T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = chunk_size
    num_chunks = len(chunk_indices)
    N = len(cu_seqlens) - 1
    device = q.device

    o = torch.empty(B, T, H, V, dtype=v.dtype, device=device)
    final_state = torch.empty(N, H, V, K, dtype=torch.float32, device=device) \
        if output_final_state else torch.empty(1, 1, 1, 1, device=device)

    ws_gcum = torch.empty(B, T, H, dtype=torch.float32, device=device)
    ws_A = torch.empty(B, T, H, BT, dtype=torch.float32, device=device)
    ws_Ai = torch.empty(B, T, H, BT, dtype=torch.float32, device=device)
    ws_w = torch.empty(B, T, H, K, dtype=torch.float32, device=device)
    ws_u = torch.empty(B, T, H, V, dtype=torch.float32, device=device)
    ws_vn = torch.empty(B, T, H, V, dtype=torch.float32, device=device)
    ws_h = torch.empty(B, num_chunks, H, V, K, dtype=torch.float32, device=device)
    h0 = initial_state.to(torch.float32).contiguous() \
        if initial_state is not None \
        else torch.zeros(1, 1, 1, 1, device=device)

    grid = (1,)
    _fused_1x1_kernel[grid](
        q, k, v, g, beta,
        o, ws_gcum, ws_A, ws_Ai,
        ws_w, ws_u, ws_vn, ws_h,
        h0, final_state,
        cu_seqlens, chunk_indices,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
        N=N, num_chunks=num_chunks, scale=scale,
        HAS_INITIAL_STATE=initial_state is not None,
        OUTPUT_FINAL_STATE=output_final_state,
        num_warps=4,
    )
    torch.cuda.synchronize()

    return o, final_state if output_final_state else None


###############################################################################
# Tests
###############################################################################
if __name__ == "__main__":
    torch.manual_seed(42)

    # Test 1: BT=16, H=2, K=32, V=32, 2 sequences
    print("=" * 60)
    print("Test 1: BT=16, H=2, K=32, V=32, 2 sequences")
    print("=" * 60)
    B, H, Hg, K, V = 1, 2, 2, 32, 32
    BT = 16
    lens = torch.tensor([17, 15], dtype=torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
    total_T = cu_seqlens[-1].item()

    q = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    k = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    v = torch.randn(B, total_T, H, V, dtype=torch.float32) * 0.1
    g = torch.randn(B, total_T, H, dtype=torch.float32) * 0.1
    beta = torch.rand(B, total_T, H, dtype=torch.float32).sigmoid()
    scale = K ** -0.5

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    o_cpu, fc, *_ = chunk_gated_delta_rule_fwd_cpu(
        q, k, v, g, beta, scale, None, False,
        cu_seqlens, chunk_indices, chunk_offsets, chunk_size=BT,
    )

    o_tr, _ = fused_chunk_gated_delta_rule_1x1(
        q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(), scale,
        None, False,
        cu_seqlens.cuda(), chunk_indices.cuda(), chunk_offsets.cuda(),
        chunk_size=BT,
    )
    o_tr_cpu = o_tr.cpu().float()

    diff = (o_cpu - o_tr_cpu).abs().max().item()
    print(f"  Max diff: {diff:.6f}")
    print(f"  CPU range: [{o_cpu.min():.6f}, {o_cpu.max():.6f}]")
    print(f"  GPU range: [{o_tr_cpu.min():.6f}, {o_tr_cpu.max():.6f}]")
    assert diff < 1e-3, f"FAIL: diff={diff}"
    print("  PASS")

    # Test 2: BT=32, H=4, K=64, V=64, 3 sequences
    print("=" * 60)
    print("Test 2: BT=32, H=4, Hg=2, K=64, V=64, 3 sequences")
    print("=" * 60)
    torch.manual_seed(123)
    B, H, Hg, K, V = 1, 4, 2, 64, 64
    BT = 32
    lens = torch.tensor([35, 67, 25], dtype=torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
    total_T = cu_seqlens[-1].item()

    q = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    k = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    v = torch.randn(B, total_T, H, V, dtype=torch.float32) * 0.1
    g = torch.randn(B, total_T, H, dtype=torch.float32) * 0.1
    beta = torch.rand(B, total_T, H, dtype=torch.float32).sigmoid()
    scale = K ** -0.5

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    o_cpu, fc, *_ = chunk_gated_delta_rule_fwd_cpu(
        q, k, v, g, beta, scale, None, True,
        cu_seqlens, chunk_indices, chunk_offsets, chunk_size=BT,
    )
    fc = fc if fc is not None else torch.zeros_like(torch.empty(3,4,64,64))

    o_tr, fs_tr = fused_chunk_gated_delta_rule_1x1(
        q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(), scale,
        None, True,
        cu_seqlens.cuda(), chunk_indices.cuda(), chunk_offsets.cuda(),
        chunk_size=BT,
    )
    o_tr_cpu = o_tr.cpu().float()

    o_diff = (o_cpu - o_tr_cpu).abs().max().item()
    fs_diff = (fc.float() - fs_tr.cpu().float()).abs().max().item()
    print(f"  o max diff: {o_diff:.6f}")
    print(f"  final_state max diff: {fs_diff:.6f}")
    print(f"  CPU o range: [{o_cpu.min():.6f}, {o_cpu.max():.6f}]")
    print(f"  GPU o range: [{o_tr_cpu.min():.6f}, {o_tr_cpu.max():.6f}]")
    assert o_diff < 1e-2, f"FAIL: o_diff={o_diff}"
    assert fs_diff < 1e-2, f"FAIL: fs_diff={fs_diff}"
    print("  PASS")

    # Test 3: BT=16, H=1, K=128, V=128, single sequence
    print("=" * 60)
    print("Test 3: BT=16, H=1, K=128, V=128, single chunk")
    print("=" * 60)
    torch.manual_seed(456)
    B, H, Hg, K, V = 1, 1, 1, 128, 128
    BT = 16
    lens = torch.tensor([16], dtype=torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
    total_T = cu_seqlens[-1].item()

    q = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    k = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    v = torch.randn(B, total_T, H, V, dtype=torch.float32) * 0.1
    g = torch.randn(B, total_T, H, dtype=torch.float32) * 0.1
    beta = torch.rand(B, total_T, H, dtype=torch.float32).sigmoid()
    scale = K ** -0.5

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    o_cpu, fc, *_ = chunk_gated_delta_rule_fwd_cpu(
        q, k, v, g, beta, scale, None, False,
        cu_seqlens, chunk_indices, chunk_offsets, chunk_size=BT,
    )

    o_tr, _ = fused_chunk_gated_delta_rule_1x1(
        q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(), scale,
        None, False,
        cu_seqlens.cuda(), chunk_indices.cuda(), chunk_offsets.cuda(),
        chunk_size=BT,
    )
    o_tr_cpu = o_tr.cpu().float()

    diff = (o_cpu - o_tr_cpu).abs().max().item()
    print(f"  Max diff: {diff:.6f}")
    print(f"  CPU range: [{o_cpu.min():.6f}, {o_cpu.max():.6f}]")
    print(f"  GPU range: [{o_tr_cpu.min():.6f}, {o_tr_cpu.max():.6f}]")
    assert diff < 5e-3, f"FAIL: diff={diff}"
    print("  PASS")

    # Test 4: BT=16, with initial_state, 2 sequences
    print("=" * 60)
    print("Test 4: BT=16, H=2, K=32, V=32, with initial_state")
    print("=" * 60)
    torch.manual_seed(789)
    B, H, Hg, K, V = 1, 2, 2, 32, 32
    BT = 16
    N = 2
    lens = torch.tensor([17, 15], dtype=torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
    total_T = cu_seqlens[-1].item()

    q = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    k = torch.randn(B, total_T, Hg, K, dtype=torch.float32) * 0.1
    v = torch.randn(B, total_T, H, V, dtype=torch.float32) * 0.1
    g = torch.randn(B, total_T, H, dtype=torch.float32) * 0.1
    beta = torch.rand(B, total_T, H, dtype=torch.float32).sigmoid()
    h0 = torch.randn(N, H, V, K, dtype=torch.float32) * 0.1
    scale = K ** -0.5

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    o_cpu, fc_cpu, *_ = chunk_gated_delta_rule_fwd_cpu(
        q, k, v, g, beta, scale, h0, True,
        cu_seqlens, chunk_indices, chunk_offsets, chunk_size=BT,
    )

    o_tr, fs_tr = fused_chunk_gated_delta_rule_1x1(
        q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(), scale,
        h0.cuda(), True,
        cu_seqlens.cuda(), chunk_indices.cuda(), chunk_offsets.cuda(),
        chunk_size=BT,
    )
    o_tr_cpu = o_tr.cpu().float()

    o_diff = (o_cpu - o_tr_cpu).abs().max().item()
    fs_diff = (fc_cpu.float() - fs_tr.cpu().float()).abs().max().item()
    print(f"  o max diff: {o_diff:.6f}")
    print(f"  final_state max diff: {fs_diff:.6f}")
    assert o_diff < 1e-3, f"FAIL: o_diff={o_diff}"
    assert fs_diff < 1e-3, f"FAIL: fs_diff={fs_diff}"
    print("  PASS")

    print("\nAll tests PASSED!")
