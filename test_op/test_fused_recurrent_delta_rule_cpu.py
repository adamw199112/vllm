"""
CPU implementation of fused_recurrent_gated_delta_rule_fwd for
fused_recurrent.py

This is a CPU reference implementation of the Triton kernel
fused_recurrent_gated_delta_rule_fwd_kernel.
"""

import torch
import math
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/adam.wang/work/vllm")


def fused_recurrent_gated_delta_rule_fwd_cpu(
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
    CPU reference implementation of the fused recurrent gated delta rule.

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
    B, T, H, K = q.shape
    HV = v.shape[2]
    V_val = v.shape[3]
    beta_is_headwise = beta.ndim == v.ndim

    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    total_T = T if cu_seqlens is None else int(cu_seqlens[-1].item())

    o = torch.zeros(B, T, HV, V_val, dtype=torch.float32)

    if inplace_final_state:
        final_state = initial_state.clone()
    else:
        final_state = torch.zeros(total_T, HV, V_val, K, dtype=torch.float32)

    for i_n in range(N):
        if cu_seqlens is not None:
            bos = cu_seqlens[i_n].item()
            eos = cu_seqlens[i_n + 1].item()
            T_seq = eos - bos
            b_idx = 0
        else:
            T_seq = T
            b_idx = i_n

        if T_seq == 0:
            continue

        # Initialize hidden state per head
        h = torch.zeros(HV, V_val, K, dtype=torch.float32)
        if initial_state is not None:
            h = initial_state[i_n].clone().float()

        for t in range(T_seq):
            if cu_seqlens is not None:
                tok = bos + t
            else:
                tok = t

            for hv in range(HV):
                i_h = hv // (HV // H)

                q_vec = q[b_idx, tok, i_h, :].float()
                k_vec = k[b_idx, tok, i_h, :].float()
                v_vec = v[b_idx, tok, hv, :].float()
                g_val = g[b_idx, tok, hv].float()
                if beta_is_headwise:
                    beta_val = beta[b_idx, tok, hv, :].float()
                else:
                    beta_val = beta[b_idx, tok, hv].float()

                if use_qk_l2norm_in_kernel:
                    q_vec = q_vec / torch.sqrt(torch.sum(q_vec * q_vec) + 1e-6)
                    k_vec = k_vec / torch.sqrt(torch.sum(k_vec * k_vec) + 1e-6)

                q_vec = q_vec * scale

                # Gated delta rule step — use elemwise ops matching Triton
                h[hv] = h[hv] * torch.exp(g_val)

                # b_v -= tl.sum(b_h * b_k[None, :], 1)
                v_new = v_vec - torch.sum(h[hv] * k_vec.unsqueeze(0), dim=1)

                # b_v *= b_beta
                v_new = v_new * beta_val

                # b_h += b_v[:, None] * b_k[None, :]
                h[hv] = h[hv] + v_new.unsqueeze(1) * k_vec.unsqueeze(0)

                # b_o = tl.sum(b_h * b_q[None, :], 1)
                o[b_idx, tok, hv, :] = torch.sum(
                    h[hv] * q_vec.unsqueeze(0), dim=1
                )

            # Store per-token hidden state (matches Triton behavior)
            if not inplace_final_state:
                final_state[tok] = h.clone()

        if inplace_final_state:
            final_state[i_n] = h

    return o, final_state


try:
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd 
    )

    HAS_TRITON = True
except Exception as e:
    print(f"Triton import failed: {e}")
    HAS_TRITON = False

class TestFusedRecurrentDeltaRule(unittest.TestCase):

    def _check_close(self, cpu_tensor, triton_tensor, atol=1e-3, rtol=1e-3):
        if triton_tensor.ndim == cpu_tensor.ndim + 1:
            triton_tensor = triton_tensor[0]
        diff = (cpu_tensor.float() - triton_tensor.float()).abs().max().item()
        mean_abs = triton_tensor.float().abs().mean().item()
        print(f"  Max diff: {diff:.6f}, mean abs ref: {mean_abs:.6f}")
        torch.testing.assert_close(
            cpu_tensor.float(), triton_tensor.float(), atol=atol, rtol=rtol
        )

    def test_basic_small(self):
        B, T, H, HV, K, V_val = 1, 8, 1, 1, 128, 128
        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=torch.float32)
        k = torch.randn(B, T, H, K, dtype=torch.float32)
        v = torch.randn(B, T, HV, V_val, dtype=torch.float32)
        g = -0.5 + torch.rand(B, T, HV, dtype=torch.float32) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32))
        scale = K ** -0.5
        h0 = torch.randn(B, HV, V_val, K, dtype=torch.float32)

        cpu_o, f_cpu = fused_recurrent_gated_delta_rule_fwd_cpu(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            scale, h0.clone(),inplace_final_state=False,
        )

        triton_o, f_triton = fused_recurrent_gated_delta_rule_fwd(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(),
            scale, h0.cuda(), inplace_final_state=False,
        )

        self._check_close(cpu_o, triton_o.cpu())
        self._check_close(f_cpu, f_triton.cpu(), atol=5e-3, rtol=5e-3)



    def test_with_l2norm(self):
        B, T, H, HV, K, V_val = 1, 16, 1, 2, 128, 128
        torch.manual_seed(789)
        q = torch.randn(B, T, H, K, dtype=torch.float32)
        k = torch.randn(B, T, H, K, dtype=torch.float32)
        v = torch.randn(B, T, HV, V_val, dtype=torch.float32)
        g = -0.5 + torch.rand(B, T, HV, dtype=torch.float32) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32))
        scale = K ** -0.5
        h0 = torch.randn(B, HV, V_val, K, dtype=torch.float32)

        cpu_o, _ = fused_recurrent_gated_delta_rule_fwd_cpu(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            scale, h0.clone(), use_qk_l2norm_in_kernel=True,
        )

        triton_o, _ = fused_recurrent_gated_delta_rule_fwd(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(),
            scale, h0.cuda(), inplace_final_state=False, use_qk_l2norm_in_kernel=True,
        )

        self._check_close(cpu_o, triton_o.cpu())

    def test_varlen(self):
        B = 1
        H = 2
        HV = 4
        K = 128
        V_val = 128
        torch.manual_seed(42)

        lens = torch.tensor([5, 3, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        q = torch.randn(B, total_T, H, K, dtype=torch.float32) 
        k = torch.randn(B, total_T, H, K, dtype=torch.float32) 
        v = torch.randn(B, total_T, HV, V_val, dtype=torch.float32) 
        g = -0.5 + torch.rand(B, total_T, HV, dtype=torch.float32) * 0.1
        beta = torch.sigmoid(torch.randn(B, total_T, HV, dtype=torch.float32))
        scale = K ** -0.5
        N = len(cu_seqlens) - 1
        h0 = torch.randn(N, HV, V_val, K, dtype=torch.float32)

        cpu_o, _ = fused_recurrent_gated_delta_rule_fwd_cpu(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            scale, h0.clone(), cu_seqlens=cu_seqlens.clone(),
        )

        triton_o, _ = fused_recurrent_gated_delta_rule_fwd(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(),
            scale, h0.cuda(), inplace_final_state=False, cu_seqlens=cu_seqlens.cuda(),
        )

        self._check_close(cpu_o, triton_o.cpu())

    def test_single_head(self):
        B, T, H, HV, K, V_val = 2, 32, 1, 1, 32, 32
        torch.manual_seed(111)
        q = torch.randn(B, T, H, K, dtype=torch.float32)
        k = torch.randn(B, T, H, K, dtype=torch.float32)
        v = torch.randn(B, T, HV, V_val, dtype=torch.float32)
        g = -0.5 + torch.rand(B, T, HV, dtype=torch.float32) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32))
        scale = K ** -0.5
        h0 = torch.randn(B, HV, V_val, K, dtype=torch.float32)

        cpu_o, _ = fused_recurrent_gated_delta_rule_fwd_cpu(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            scale, h0.clone(),
        )

        triton_o, _ = fused_recurrent_gated_delta_rule_fwd(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(),
            scale, h0.cuda(), inplace_final_state=False,
        )

        self._check_close(cpu_o, triton_o.cpu())

    def test_no_initial_state(self):
        B, T, H, HV, K, V_val = 1, 16, 2, 2, 128, 128
        torch.manual_seed(222)
        q = torch.randn(B, T, H, K, dtype=torch.float32) 
        k = torch.randn(B, T, H, K, dtype=torch.float32) 
        v = torch.randn(B, T, HV, V_val, dtype=torch.float32) 
        g = -0.5 + torch.rand(B, T, HV, dtype=torch.float32) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32))
        scale = K ** -0.5
        h0 = torch.zeros(B, HV, V_val, K, dtype=torch.float32)

        cpu_o, _ = fused_recurrent_gated_delta_rule_fwd_cpu(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            scale, h0.clone(),
        )

        triton_o, _ = fused_recurrent_gated_delta_rule_fwd(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(),
            scale, h0.cuda(), inplace_final_state=False,
        )

        self._check_close(cpu_o, triton_o.cpu())

    def test_varlen_headwise_beta(self):
        B = 1
        H = 2
        HV = 4
        K = 128
        V_val = 128
        torch.manual_seed(555)

        lens = torch.tensor([4, 6, 3], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        q = torch.randn(B, total_T, H, K, dtype=torch.float32) 
        k = torch.randn(B, total_T, H, K, dtype=torch.float32) 
        v = torch.randn(B, total_T, HV, V_val, dtype=torch.float32) 
        g = -0.5 + torch.rand(B, total_T, HV, dtype=torch.float32) * 0.1
        beta = torch.sigmoid(torch.randn(B, total_T, HV, V_val, dtype=torch.float32))
        scale = K ** -0.5
        N = len(cu_seqlens) - 1
        h0 = torch.randn(N, HV, V_val, K, dtype=torch.float32)

        cpu_o, _ = fused_recurrent_gated_delta_rule_fwd_cpu(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            scale, h0.clone(), cu_seqlens=cu_seqlens.clone(),
        )

        triton_o, _ = fused_recurrent_gated_delta_rule_fwd(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(),
            scale, h0.cuda(), inplace_final_state=False, cu_seqlens=cu_seqlens.cuda(),
        )

        self._check_close(cpu_o, triton_o.cpu())

    def test_large_k_128(self):
        B, T, H, HV, K, V_val = 1, 8, 1, 1, 128, 128
        torch.manual_seed(999)
        q = torch.randn(B, T, H, K, dtype=torch.float32)
        k = torch.randn(B, T, H, K, dtype=torch.float32)
        v = torch.randn(B, T, HV, V_val, dtype=torch.float32)
        g = -0.5 + torch.rand(B, T, HV, dtype=torch.float32) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32))
        scale = K ** -0.5
        h0 = torch.randn(B, HV, V_val, K, dtype=torch.float32) * 0.01

        cpu_o, _ = fused_recurrent_gated_delta_rule_fwd_cpu(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            scale, h0.clone(),
        )

        triton_o, _ = fused_recurrent_gated_delta_rule_fwd(
            q.cuda(), k.cuda(), v.cuda(), g.cuda(), beta.cuda(),
            scale, h0.cuda(), inplace_final_state=False,
        )

        self._check_close(cpu_o, triton_o.cpu())


if __name__ == "__main__":
    unittest.main(verbosity=2)
