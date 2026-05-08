"""
CPU implementation of chunk_gated_delta_rule_fwd_h for chunk_delta_h.py

This is a CPU implementation of the Triton kernel chunk_gated_delta_rule_fwd_h.
It supports:
- Basic chunked delta rule computation (no gates)
- Multiple heads (Hg=H)

Note: This is a reference implementation. The exact output may differ slightly from 
the Triton kernel due to parallel vs sequential processing differences.
"""

import torch
import math
import sys


def exp(x):
    """Exponential function"""
    return torch.exp(x)


def chunk_gated_delta_rule_fwd_h_cpu(
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
    """
    CPU implementation of chunked gated delta rule forward pass.
    
    This function computes:
        v_new = v - w @ h
        h = h + k.T @ v_new
        
    Where h is maintained across chunks in a chunked recurrence.
    
    Args:
        k: [B, T, Hg, K] key tensor
        w: [B, T, H, K] weight tensor  
        u: [B, T, H, V] value tensor (input)
        g: [B, T, H] optional gate (decay)
        gk: [B, T, H, K] optional gate for key
        initial_state: [N, H, V, K] optional initial state
        output_final_state: whether to output final state
        chunk_size: chunk size for computation
        save_new_value: whether to compute and return v_new
        cu_seqlens: cumulative seq lengths for variable length
        chunk_indices: chunk indices
        chunk_offsets: chunk offsets
        
    Returns:
        h: [B, NT, H, V, K] hidden states per chunk
        v_new: [B, T, H, V] updated values (if save_new_value=True)
        final_state: [N, H, V, K] final state (if output_final_state=True)
    """
    B, T, Hg, K = k.shape
    H = u.shape[-2]
    BT = chunk_size
    V = u.shape[-1]

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    NT = math.ceil(T / BT)

    h = torch.empty(B, NT, H, V, K, dtype=torch.float32, device=k.device)
    final_state = torch.empty(N, H, V, K, dtype=torch.float32, device=k.device) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None

    for b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            cur_BT = t_end - t_start
            
            last_idx = t_end - 1

            for h_idx in range(H):
                k_head_idx = h_idx // (H // Hg) if H != Hg else h_idx

                h_state_cur = torch.zeros(V, K, dtype=torch.float32, device=k.device)
                if initial_state is not None:
                    h_state_cur = initial_state[b * H + h_idx].clone()

                h[b, i_t, h_idx] = h_state_cur.clone()

                for t in range(cur_BT):
                    t_idx = t_start + t
                    v_slice = u[b, t_idx, h_idx, :]
                    w_slice = w[b, t_idx, h_idx, :]

                    v_out_val = v_slice - torch.mm(w_slice.unsqueeze(0), h_state_cur.T).squeeze(0)

                    if g is not None:
                        m_t = (t_start + t) < T
                        if m_t and t > 0:
                            g_last = g[b, t_start + t - 1, h_idx]
                            g_curr = g[b, t_idx, h_idx]
                            v_out_val = v_out_val * exp(g_last - g_curr)
                        
                        g_last_decay = exp(g[b, t_idx, h_idx])
                        h_state_cur = h_state_cur * g_last_decay

                    if gk is not None:
                        gk_slice = gk[b, t_idx, h_idx, :]
                        h_state_cur = h_state_cur * exp(gk_slice)

                    k_slice = k[b, t_idx, k_head_idx, :]
                    h_update = torch.outer(k_slice, v_out_val).T
                    h_state_cur = h_state_cur + h_update
            
            if save_new_value:
                for h_idx in range(H):
                    h_start = h[b, i_t, h_idx].clone()
                    w_chunk = w[b, t_start:t_end, h_idx, :]
                    u_chunk = u[b, t_start:t_end, h_idx, :]
                    
                    v_new_chunk = u_chunk - torch.mm(w_chunk, h_start.T)
                    
                    if g is not None:
                        for t in range(cur_BT):
                            if t > 0:
                                g_last = g[b, t_start + t - 1, h_idx]
                                g_curr = g[b, t_start + t, h_idx]
                                v_new_chunk[t] = v_new_chunk[t] * exp(g_last - g_curr)
                    
                    v_new[b, t_start:t_end, h_idx, :] = v_new_chunk

    return h, v_new, final_state



import unittest

sys.path.insert(0, "/home/adam.wang/work/vllm")
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h as chunk_gated_delta_rule_fwd_h_triton
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices


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
                        v_out_val = v_out_val * exp(g_last - g_curr)
                    
                    g_last_decay = exp(g[0, t_idx, h_idx])
                    h_state_cur = h_state_cur * g_last_decay

                if gk is not None:
                    gk_slice = gk[0, t_idx, h_idx, :]
                    h_state_cur = h_state_cur * exp(gk_slice)

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
                            v_new_val = v_new_val * exp(g_last - g_curr)
                    
                    v_new[0, t_idx, h_idx, :] = v_new_val

    return h, v_new, final_state


class TestCPU(unittest.TestCase):
    def test_basic(self):
        B, T, Hg, K, H, V = 1, 32, 1, 16, 1, 8
        BT = 32
        torch.manual_seed(42)
        k = torch.randn(B, T, Hg, K) * 0.1
        w = torch.randn(B, T, H, K) * 0.1
        u = torch.randn(B, T, H, V) * 0.1

        h_cpu, v_new_cpu, _ = chunk_gated_delta_rule_fwd_h_cpu(k, w, u, chunk_size=BT)
        h_triton, v_new_triton, _ = chunk_gated_delta_rule_fwd_h_triton(
            k=k.cuda(), w=w.cuda(), u=u.cuda(), chunk_size=BT,
        )

        print(f"h diff: {(h_cpu.float() - h_triton.cpu().float()).abs().max().item():.6f}")
        print(f"v_new diff: {(v_new_cpu.float() - v_new_triton.cpu().float()).abs().max().item():.6f}")
        torch.testing.assert_close(v_new_cpu.float(), v_new_triton.cpu().float(), atol=1e-2, rtol=1e-2)

    def test_varlen_16(self):
        print("\n=== Varlen BT=16 ===")
        B = 1
        H = 2
        Hg = 2
        K = 16
        V = 8
        BT = 16
        torch.manual_seed(42)

        lens = torch.tensor([5, 3, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        k = torch.randn(B, total_T, Hg, K) * 0.1
        w = torch.randn(B, total_T, H, K) * 0.1
        u = torch.randn(B, total_T, H, V) * 0.1

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        h_cpu, v_new_cpu, _ = chunk_gated_delta_rule_fwd_h_cpu_varlen(
            k, w, u, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        h_triton, v_new_triton, _ = chunk_gated_delta_rule_fwd_h_triton(
            k=k.cuda(), w=w.cuda(), u=u.cuda(), chunk_size=BT,
            cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda(),
        )

        h_diff = (h_cpu.float() - h_triton.cpu().float()).abs().max().item()
        v_diff = (v_new_cpu.float() - v_new_triton.cpu().float()).abs().max().item()
        print(f"Varlen BT=16 - h max diff: {h_diff:.6f}, v_new max diff: {v_diff:.6f}")
        torch.testing.assert_close(v_new_cpu.float(), v_new_triton.cpu().float(), atol=1e-2, rtol=1e-2)

    def test_varlen_32(self):
        print("\n=== Varlen BT=32 ===")
        B = 1
        H = 4
        Hg = 2
        K = 32
        V = 16
        BT = 32
        torch.manual_seed(123)

        lens = torch.tensor([10, 15, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        k = torch.randn(B, total_T, Hg, K) * 0.1
        w = torch.randn(B, total_T, H, K) * 0.1
        u = torch.randn(B, total_T, H, V) * 0.1

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        h_cpu, v_new_cpu, _ = chunk_gated_delta_rule_fwd_h_cpu_varlen(
            k, w, u, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        h_triton, v_new_triton, _ = chunk_gated_delta_rule_fwd_h_triton(
            k=k.cuda(), w=w.cuda(), u=u.cuda(), chunk_size=BT,
            cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda(),
        )

        h_diff = (h_cpu.float() - h_triton.cpu().float()).abs().max().item()
        v_diff = (v_new_cpu.float() - v_new_triton.cpu().float()).abs().max().item()
        print(f"Varlen BT=32 - h max diff: {h_diff:.6f}, v_new max diff: {v_diff:.6f}")
        torch.testing.assert_close(v_new_cpu.float(), v_new_triton.cpu().float(), atol=1e-2, rtol=1e-2)

    def test_varlen_64(self):
        print("\n=== Varlen BT=64 ===")
        B = 1
        H = 4
        Hg = 2
        K = 128
        V = 128
        BT = 64
        torch.manual_seed(456)

        lens = torch.tensor([20, 15, 33], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        k = torch.randn(B, total_T, Hg, K) * 0.1
        w = torch.randn(B, total_T, H, K) * 0.1
        u = torch.randn(B, total_T, H, V) * 0.1

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        h_cpu, v_new_cpu, _ = chunk_gated_delta_rule_fwd_h_cpu_varlen(
            k, w, u, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        h_triton, v_new_triton, _ = chunk_gated_delta_rule_fwd_h_triton(
            k=k.cuda(), w=w.cuda(), u=u.cuda(), chunk_size=BT,
            cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda(),
        )

        h_diff = (h_cpu.float() - h_triton.cpu().float()).abs().max().item()
        v_diff = (v_new_cpu.float() - v_new_triton.cpu().float()).abs().max().item()
        print(f"Varlen BT=64 - h max diff: {h_diff:.6f}, v_new max diff: {v_diff:.6f}")
        torch.testing.assert_close(v_new_cpu.float(), v_new_triton.cpu().float(), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    unittest.main(verbosity=2)