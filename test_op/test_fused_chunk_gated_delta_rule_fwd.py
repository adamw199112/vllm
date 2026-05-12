"""
Test case for chunk_gated_delta_rule_fwd (ChunkGatedDeltaRuleFunction).

This test composes individual CPU reference implementations from test_op/
to validate the full pipeline against the Triton implementation.

All test cases explicitly provide cu_seqlens, chunk_indices, and chunk_offsets (never None).
"""

import torch
import math
import unittest
import sys

sys.path.insert(0, "/home/adam.wang/work/vllm")


# --- CPU sub-function imports ---
def _exp(x):
    return torch.exp(x)


# 1. chunk_local_cumsum CPU
def chunk_local_cumsum_cpu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = None,
):
    B, T, H = g.shape
    if output_dtype is None:
        output_dtype = g.dtype
    o = torch.empty_like(g, dtype=output_dtype)

    for idx in range(len(chunk_indices)):
        i_n = int(chunk_indices[idx, 0].item())
        i_t = int(chunk_indices[idx, 1].item())
        bos = int(cu_seqlens[i_n].item())
        eos = int(cu_seqlens[i_n + 1].item())
        T_seq = eos - bos
        t0 = i_t * chunk_size
        t1 = min(t0 + chunk_size, T_seq)
        if t0 >= T_seq:
            continue
        for h in range(H):
            chunk = g[0, bos + t0 : bos + t1, h].to(torch.float32)
            cumsum = torch.cumsum(chunk, dim=0)
            if reverse:
                S = chunk.sum()
                cumsum = -cumsum + S + chunk
            o[0, bos + t0 : bos + t1, h] = cumsum.to(output_dtype)
    return o


# 2. chunk_scaled_dot_kkt CPU
def chunk_scaled_dot_kkt_cpu(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        NT = math.ceil(T / BT)
    else:
        NT = len(chunk_indices)

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)

    for idx in range(len(chunk_indices)):
        i_n = int(chunk_indices[idx, 0].item())
        i_t = int(chunk_indices[idx, 1].item())
        bos = int(cu_seqlens[i_n].item())
        eos = int(cu_seqlens[i_n + 1].item())
        T_seq = eos - bos
        chunk_start = i_t * BT
        chunk_end = min(chunk_start + BT, T_seq)
        cur_BT = chunk_end - chunk_start
        if cur_BT <= 0:
            continue
        for b in range(B):
            for h in range(H):
                k_head_idx = h // (H // Hg) if H != Hg else h
                beta_slice = beta[b, bos + chunk_start : bos + chunk_end, h]
                k_slice = k[b, bos + chunk_start : bos + chunk_end, k_head_idx, :]
                k_beta = k_slice * beta_slice.unsqueeze(1)
                A_chunk = torch.matmul(k_beta, k_slice.T)
                if g is not None:
                    g_slice = g[b, bos + chunk_start : bos + chunk_end, h]
                    g_diff = g_slice.unsqueeze(1) - g_slice.unsqueeze(0)
                    A_chunk = A_chunk * _exp(g_diff)
                causal_mask = torch.tril(
                    torch.ones(cur_BT, cur_BT, device=k.device), diagonal=-1
                )
                A_chunk = A_chunk * causal_mask
                A[b, bos + chunk_start : bos + chunk_end, h, :cur_BT] = A_chunk
    return A


# 3. solve_tril CPU (varlen)
def solve_tril_cpu_varlen(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    BT: int,
) -> torch.Tensor:
    B, total_T, H, _ = A.shape
    Ai = torch.zeros_like(A)
    num_chunks = len(chunk_indices)
    for idx in range(num_chunks):
        i_n = int(chunk_indices[idx, 0].item())
        i_t = int(chunk_indices[idx, 1].item())
        bos = int(cu_seqlens[i_n].item())
        eos = int(cu_seqlens[i_n + 1].item())
        T_seq = eos - bos
        chunk_start = i_t * BT
        chunk_end = min(chunk_start + BT, T_seq)
        cur_BT = chunk_end - chunk_start
        if cur_BT == 0:
            continue
        for h in range(H):
            A_chunk = A[0, bos + chunk_start : bos + chunk_end, h, :cur_BT].clone()
            M = torch.eye(cur_BT, dtype=A.dtype) + torch.tril(A_chunk)
            X = torch.zeros(cur_BT, cur_BT, dtype=A.dtype)
            for j in range(cur_BT):
                X[j, j] = 1.0
                for i in range(j + 1, cur_BT):
                    s = torch.dot(M[i, :i], X[:i, j])
                    X[i, j] = -s / M[i, i]
            Ai[0, bos + chunk_start : bos + chunk_end, h, :cur_BT] = X
    return Ai


# 4. recompute_w_u CPU (varlen)
def recompute_w_u_cpu_varlen(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, total_T, Hg, K = k.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = A.shape[-1]

    u = k.new_empty(B, total_T, H, V, dtype=k.dtype)
    w = k.new_empty(B, total_T, H, K, dtype=k.dtype)

    num_chunks = len(chunk_indices)
    BK = 64
    BV = 64

    for idx in range(num_chunks):
        i_n = int(chunk_indices[idx, 0].item())
        i_t = int(chunk_indices[idx, 1].item())
        bos = int(cu_seqlens[i_n].item())
        eos = int(cu_seqlens[i_n + 1].item())
        T_seq = eos - bos
        chunk_start = i_t * BT
        chunk_end = min(chunk_start + BT, T_seq)
        cur_BT = chunk_end - chunk_start
        if cur_BT == 0:
            continue
        for h in range(H):
            k_head_idx = h // (H // Hg) if H != Hg else h
            beta_slice = beta[0, bos + chunk_start : bos + chunk_end, h]
            A_slice = A[0, bos + chunk_start : bos + chunk_end, h, :cur_BT]
            g_slice = _exp(g_cumsum[0, bos + chunk_start : bos + chunk_end, h])
            for i_v in range(math.ceil(V / BV)):
                v_start = i_v * BV
                v_end = min(v_start + BV, V)
                v_slice = v[0, bos + chunk_start : bos + chunk_end, h, v_start:v_end]
                v_scaled = v_slice * beta_slice[:, None]
                u_slice = torch.matmul(A_slice[:cur_BT, :cur_BT], v_scaled)
                u[0, bos + chunk_start : bos + chunk_end, h, v_start:v_end] = u_slice
            for i_k in range(math.ceil(K / BK)):
                k_start = i_k * BK
                k_end = min(k_start + BK, K)
                k_slice = k[0, bos + chunk_start : bos + chunk_end, k_head_idx, k_start:k_end]
                k_scaled = k_slice * beta_slice[:, None] * g_slice[:, None]
                w_slice = torch.matmul(A_slice[:cur_BT, :cur_BT], k_scaled)
                w[0, bos + chunk_start : bos + chunk_end, h, k_start:k_end] = w_slice
    return w, u


# 5. chunk_gated_delta_rule_fwd_h CPU (varlen)
def chunk_gated_delta_rule_fwd_h_cpu_varlen(
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
):
    B, total_T, Hg, K = k.shape
    H = u.shape[-2]
    BT = chunk_size
    V = u.shape[-1]

    N = len(cu_seqlens) - 1
    num_chunks = len(chunk_indices)

    h = torch.empty(B, num_chunks, H, V, K, dtype=torch.float32, device=k.device)
    final_state = torch.empty(N, H, V, K, dtype=torch.float32, device=k.device) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    for idx in range(num_chunks):
        i_n = chunk_indices[idx, 0].item()
        i_t = chunk_indices[idx, 1].item()
        bos = cu_seqlens[i_n].item()
        eos = cu_seqlens[i_n + 1].item()
        T_seq = eos - bos
        t_start = i_t * BT
        t_end = min(t_start + BT, T_seq)
        cur_BT = t_end - t_start
        if cur_BT == 0:
            continue
        for h_idx in range(H):
            k_head_idx = h_idx // (H // Hg) if H != Hg else h_idx
            h_state_cur = torch.zeros(V, K, dtype=torch.float32, device=k.device)
            if initial_state is not None:
                h_state_cur = initial_state[i_n * H + h_idx].clone()
            h[0, idx, h_idx] = h_state_cur.clone()
            for t in range(cur_BT):
                t_idx = bos + t_start + t
                v_slice = u[0, t_idx, h_idx, :]
                w_slice = w[0, t_idx, h_idx, :]
                v_out_val = v_slice - torch.mm(w_slice.unsqueeze(0), h_state_cur.T).squeeze(0)
                if g is not None:
                    if t > 0:
                        g_last = g[0, t_idx - 1, h_idx]
                        g_curr = g[0, t_idx, h_idx]
                        v_out_val = v_out_val * _exp(g_last - g_curr)
                    g_last_decay = _exp(g[0, t_idx, h_idx])
                    h_state_cur = h_state_cur * g_last_decay
                if gk is not None:
                    gk_slice = gk[0, t_idx, h_idx, :]
                    h_state_cur = h_state_cur * _exp(gk_slice)
                k_slice = k[0, t_idx - bos, k_head_idx, :]
                h_update = torch.outer(k_slice, v_out_val).T
                h_state_cur = h_state_cur + h_update

        if save_new_value:
            for h_idx in range(H):
                h_start = h[0, idx, h_idx].clone()
                for t in range(cur_BT):
                    t_idx = bos + t_start + t
                    v_slice = u[0, t_idx, h_idx, :]
                    w_slice = w[0, t_idx, h_idx, :]
                    v_new_val = v_slice - torch.mm(w_slice.unsqueeze(0), h_start.T).squeeze(0)
                    if g is not None:
                        if t > 0:
                            g_last = g[0, t_idx - 1, h_idx]
                            g_curr = g[0, t_idx, h_idx]
                            v_new_val = v_new_val * _exp(g_last - g_curr)
                    v_new[0, t_idx, h_idx, :] = v_new_val
    return h, v_new, final_state


# 6. chunk_fwd_o CPU (varlen)
def chunk_fwd_o_cpu_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    B, total_T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = chunk_size
    num_chunks = len(chunk_indices)
    if scale is None:
        scale = K ** -0.5
    o = torch.empty_like(v)
    for idx in range(num_chunks):
        i_n = chunk_indices[idx, 0].item()
        i_t = chunk_indices[idx, 1].item()
        bos = cu_seqlens[i_n].item()
        eos = cu_seqlens[i_n + 1].item()
        T_seq = eos - bos
        t_start = i_t * BT
        t_end = min(t_start + BT, T_seq)
        cur_BT = t_end - t_start
        if cur_BT == 0:
            continue
        for h_idx in range(H):
            k_head_idx = h_idx // (H // Hg) if H != Hg else h_idx
            q_chunk = q[0, bos + t_start : bos + t_end, k_head_idx, :]
            k_chunk = k[0, bos + t_start : bos + t_end, k_head_idx, :]
            v_chunk = v[0, bos + t_start : bos + t_end, h_idx, :]
            h_state = h[0, idx, h_idx]
            o_chunk = torch.matmul(q_chunk, h_state.T)
            A_chunk = torch.matmul(q_chunk, k_chunk.T)
            if g is not None:
                g_chunk = g[0, bos + t_start : bos + t_end, h_idx]
                o_chunk = o_chunk * _exp(g_chunk[:, None])
                A_chunk = A_chunk * _exp(g_chunk[:, None] - g_chunk[None, :])
            mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=torch.bool, device=q.device))
            A_chunk = A_chunk.masked_fill(~mask, 0)
            o_chunk = o_chunk + torch.matmul(A_chunk, v_chunk)
            o_chunk = o_chunk * scale
            o[0, bos + t_start : bos + t_end, h_idx, :] = o_chunk
    return o


# --- Full pipeline CPU (fused: steps 2-6 share one chunk loop) ---
def chunk_gated_delta_rule_fwd_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int = 64,
):
    B, T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = chunk_size
    num_chunks = len(chunk_indices)
    N = len(cu_seqlens) - 1

    # Step 1: chunk_local_cumsum (needs separate pass for correctness)
    g_cumsum = chunk_local_cumsum_cpu(
        g, chunk_size=chunk_size, reverse=False,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, head_first=False,
    )

    # Allocate outputs
    A = torch.zeros(B, T, H, BT, dtype=torch.float32)
    Ai = torch.zeros_like(A)
    w = torch.empty(B, T, H, K, dtype=k.dtype)
    u = torch.empty(B, T, H, V, dtype=k.dtype)
    h = torch.empty(B, num_chunks, H, V, K, dtype=torch.float32)
    v_new = torch.empty(B, T, H, V, dtype=k.dtype)
    final_state = torch.empty(N, H, V, K, dtype=torch.float32) 
    o = torch.empty(B, T, H, V, dtype=v.dtype)

    # Track running state per (sequence, head) across chunks
    

    # Fused loop: steps 2-6 per chunk
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

            # --- Step 2: chunk_scaled_dot_kkt ---
            beta_slice = beta[0, bos + t_start : bos + t_end, h_idx]
            k_slice = k[0, bos + t_start : bos + t_end, k_head_idx, :]
            k_beta = k_slice * beta_slice.unsqueeze(1)
            A_chunk = torch.matmul(k_beta, k_slice.T)
            g_slice = g_cumsum[0, bos + t_start : bos + t_end, h_idx]
            g_diff = g_slice.unsqueeze(1) - g_slice.unsqueeze(0)
            A_chunk = A_chunk * _exp(g_diff)
            causal_mask = torch.tril(torch.ones(cur_BT, cur_BT), diagonal=-1)
            A_chunk = A_chunk * causal_mask
            A[0, bos + t_start : bos + t_end, h_idx, :cur_BT] = A_chunk

            # --- Step 3: solve_tril ---
            M = torch.eye(cur_BT) + torch.tril(A_chunk)
            X = torch.zeros(cur_BT, cur_BT)
            for j in range(cur_BT):
                X[j, j] = 1.0
                for i in range(j + 1, cur_BT):
                    X[i, j] = -torch.dot(M[i, :i], X[:i, j]) / M[i, i]
            Ai[0, bos + t_start : bos + t_end, h_idx, :cur_BT] = X

            # --- Step 4: recompute_w_u ---
            A_slice = X
            g_exp = _exp(g_slice)
            for i_v in range(math.ceil(V / 64)):
                v_s, v_e = i_v * 64, min(i_v * 64 + 64, V)
                v_sliced = v[0, bos + t_start : bos + t_end, h_idx, v_s:v_e]
                u[0, bos + t_start : bos + t_end, h_idx, v_s:v_e] = torch.matmul(A_slice, v_sliced * beta_slice[:, None])
            for i_k in range(math.ceil(K / 64)):
                k_s, k_e = i_k * 64, min(i_k * 64 + 64, K)
                k_sliced = k[0, bos + t_start : bos + t_end, k_head_idx, k_s:k_e]
                w[0, bos + t_start : bos + t_end, h_idx, k_s:k_e] = torch.matmul(A_slice, k_sliced * beta_slice[:, None] * g_exp[:, None])

            # --- Step 5: chunk_gated_delta_rule_fwd_h (chunked gating matching Triton) ---
            h_start = initial_state[i_n * H + h_idx].clone() if initial_state is not None else torch.zeros(V, K)
            h[0, idx, h_idx] = h_start.clone()

            last_idx_in_chunk = t_end - 1
            g_last = g_cumsum[0, bos + last_idx_in_chunk, h_idx]

            # Gate previous state: h *= exp(g_last) (BEFORE accumulating contributions)
            h_state_cur = h_start * _exp(g_last)

            # Compute v_new (without gating) and accumulate state (with chunked gating)
            for t in range(cur_BT):
                t_idx = bos + t_start + t
                k_head_idx_t = h_idx // (H // Hg) if H != Hg else h_idx
                v_slice = u[0, t_idx, h_idx, :]
                w_slice = w[0, t_idx, h_idx, :]
                # v_new = u - w @ h_start (no gating, matching Triton SAVE_NEW_VALUE)
                v_new_val = v_slice - torch.mm(w_slice.unsqueeze(0), h_start.T).squeeze(0)
                v_new[0, t_idx, h_idx, :] = v_new_val
                # Gated v for state update: v *= exp(g_last - g_curr)
                g_curr = g_cumsum[0, t_idx, h_idx]
                v_gated = v_new_val * _exp(g_last - g_curr)
                k_t = k[0, t_idx, k_head_idx_t, :]
                h_state_cur += torch.outer(k_t, v_gated).T

            final_state[i_n, h_idx] = h_state_cur.clone()

            # --- Step 6: chunk_fwd_o (same h_idx iteration) ---
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


# --- Triton imports ---
from vllm.model_executor.layers.fla.ops.chunk import (
    chunk_gated_delta_rule,
    ChunkGatedDeltaRuleFunction,
)
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE


class TestChunkGatedDeltaRuleFwd(unittest.TestCase):
    """Test ChunkGatedDeltaRuleFunction full pipeline with explicit chunk_offsets."""

    def _run_cpu_pipeline(
        self, q, k, v, g, beta, scale, initial_state, output_final_state,
        cu_seqlens, chunk_indices, chunk_offsets, chunk_size,
    ):
        return chunk_gated_delta_rule_fwd_cpu(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets, chunk_size=chunk_size,
        )

    def _run_triton_pipeline(
        self, q, k, v, g, beta, scale, initial_state, output_final_state,
        cu_seqlens, chunk_indices, chunk_offsets,
    ):
        return chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

    def test_varlen_full_pipeline_bt16(self):
        print("\n=== Full pipeline varlen BT=16 with explicit chunk_offsets ===")
        B, H, Hg, K, V = 1, 2, 2, 64, 64
        BT = 16
        torch.manual_seed(42)

        lens = torch.tensor([5, 3, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        q = torch.randn(B, total_T, Hg, K, dtype=torch.bfloat16) * 0.1
        k = torch.randn(B, total_T, Hg, K, dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, total_T, H, V, dtype=torch.bfloat16) * 0.1
        g = torch.randn(B, total_T, H, dtype=torch.bfloat16) * 0.1
        beta = torch.rand(B, total_T, H, dtype=torch.bfloat16).sigmoid()
        scale = K ** -0.5

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

        o_cpu, final_state_cpu, *_ = self._run_cpu_pipeline(
            q.float(), k.float(), v.float(), g.float(), beta.float(), scale,
            None, False, cu_seqlens, chunk_indices, chunk_offsets, chunk_size=BT,
        )

        o_triton, final_state_triton = self._run_triton_pipeline(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(), scale,
            None, False, cu_seqlens.cuda(), chunk_indices.cuda(), chunk_offsets.cuda(),
        )

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"BT=16 - o max diff: {o_diff:.6f}")
        torch.testing.assert_close(o_cpu.float(), o_triton.cpu().float(), atol=1e-2, rtol=1e-2)
        self.assertIsNone(final_state_triton)

    def test_varlen_full_pipeline_bt32(self):
        print("\n=== Full pipeline varlen BT=32 with explicit chunk_offsets ===")
        B, H, Hg, K, V = 1, 4, 2, 128, 128
        BT = 32
        torch.manual_seed(123)

        lens = torch.tensor([10, 15, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        q = torch.randn(B, total_T, Hg, K, dtype=torch.bfloat16) * 0.1
        k = torch.randn(B, total_T, Hg, K, dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, total_T, H, V, dtype=torch.bfloat16) * 0.1
        g = torch.randn(B, total_T, H, dtype=torch.bfloat16) * 0.1
        beta = torch.rand(B, total_T, H, dtype=torch.bfloat16).sigmoid()
        scale = K ** -0.5

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

        o_cpu, final_state_cpu, *_ = self._run_cpu_pipeline(
            q.float(), k.float(), v.float(), g.float(), beta.float(), scale,
            None, True, cu_seqlens, chunk_indices, chunk_offsets, chunk_size=BT,
        )

        o_triton, final_state_triton = self._run_triton_pipeline(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(), scale,
            None, True, cu_seqlens.cuda(), chunk_indices.cuda(), chunk_offsets.cuda(),
        )

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"BT=32 - o max diff: {o_diff:.6f}")
        print("===final_state_triton shape====")
        print(final_state_triton.cpu().shape)
        f_diff = (final_state_cpu.float() - final_state_triton.cpu().float()).abs().max().item()
        print(f"BT=32 - final state max diff: {f_diff:.6f}")
        torch.testing.assert_close(o_cpu.float(), o_triton.cpu().float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(final_state_cpu.float(), final_state_triton.cpu().float(), atol=1e-2, rtol=1e-2)

    def test_varlen_full_pipeline_bt64(self):
        print("\n=== Full pipeline varlen BT=64 with explicit chunk_offsets ===")
        B, H, Hg, K, V = 1, 4, 2, 64, 64
        BT = 64
        torch.manual_seed(456)

        lens = torch.tensor([20, 15, 33], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        q = torch.randn(B, total_T, Hg, K, dtype=torch.bfloat16) * 0.1
        k = torch.randn(B, total_T, Hg, K, dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, total_T, H, V, dtype=torch.bfloat16) * 0.1
        g = torch.randn(B, total_T, H, dtype=torch.bfloat16) * 0.1
        beta = torch.rand(B, total_T, H, dtype=torch.bfloat16).sigmoid()
        scale = K ** -0.5

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

        o_cpu, final_state_cpu, *_ = self._run_cpu_pipeline(
            q.float(), k.float(), v.float(), g.float(), beta.float(), scale,
            None, True, cu_seqlens, chunk_indices, chunk_offsets, chunk_size=BT,
        )

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        o_triton, final_state_triton = self._run_triton_pipeline(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(), scale,
            None, True, cu_seqlens.cuda(), chunk_indices.cuda(), chunk_offsets.cuda(),
        )
        torch.cuda.synchronize()

        print("===final_state_triton shape====")
        print(final_state_triton.cpu().shape)
        f_diff = (final_state_cpu.float() - final_state_triton.cpu().float()).abs().max().item()
        print(f"BT=64 - final state max diff: {f_diff:.6f}")

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"BT=64 - o max diff: {o_diff:.6f}")
        torch.testing.assert_close(o_cpu.float(), o_triton.cpu().float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(final_state_cpu.float(), final_state_triton.cpu().float(), atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
