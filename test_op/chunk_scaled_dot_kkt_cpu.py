# SPDX-License-Identifier: Apache-2.0
# CPU implementation of chunk_scaled_dot_kkt
# This computes: beta * K * K^T with causal masking

import torch
import math


def prepare_chunk_indices_cpu(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Prepare chunk indices for variable length sequences."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    result = torch.stack([torch.cumsum(indices.eq(0).int(), 0) - 1, indices], 1)
    return result.to(cu_seqlens.dtype)


def chunk_scaled_dot_kkt_cpu(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute beta * K * K^T.

    Args:
        k: The key tensor of shape [B, T, Hg, K]
        g: The cumulative sum of the gate tensor of shape [B, T, H]
        beta: The beta tensor of shape [B, T, H]
        cu_seqlens: The cumulative sequence lengths
        chunk_indices: Pre-computed chunk indices
        chunk_size: The chunk size
        output_dtype: The output dtype

    Returns:
        beta * K * K^T of shape [B, T, H, BT]
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices_cpu(cu_seqlens, BT)

    if cu_seqlens is None:
        NT = math.ceil(T / BT)
    else:
        NT = len(chunk_indices)

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)

    if cu_seqlens is None:
        for i_t in range(NT):
            chunk_start = i_t * BT
            chunk_end = min(chunk_start + BT, T)
            cur_BT = chunk_end - chunk_start

            for b in range(B):
                for h in range(H):
                    k_head_idx = h // (H // Hg)

                    beta_slice = beta[b, chunk_start:chunk_end, h]
                    k_slice = k[b, chunk_start:chunk_end, k_head_idx, :]

                    k_beta = k_slice * beta_slice.unsqueeze(1)
                    A_chunk = torch.matmul(k_beta, k_slice.T)

                    if g is not None:
                        g_slice = g[b, chunk_start:chunk_end, h]
                        g_diff = g_slice.unsqueeze(1) - g_slice.unsqueeze(0)
                        A_chunk = A_chunk * torch.exp(g_diff)

                    causal_mask = torch.tril(
                        torch.ones(cur_BT, cur_BT, device=k.device), diagonal=-1
                    )
                    A_chunk = A_chunk * causal_mask

                    A[b, chunk_start:chunk_end, h, :cur_BT] = A_chunk
    else:
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
                    k_head_idx = h // (H // Hg)

                    beta_slice = beta[b, bos + chunk_start : bos + chunk_end, h]
                    k_slice = k[b, bos + chunk_start : bos + chunk_end, k_head_idx, :]

                    k_beta = k_slice * beta_slice.unsqueeze(1)
                    A_chunk = torch.matmul(k_beta, k_slice.T)

                    if g is not None:
                        g_slice = g[b, bos + chunk_start : bos + chunk_end, h]
                        g_diff = g_slice.unsqueeze(1) - g_slice.unsqueeze(0)
                        A_chunk = A_chunk * torch.exp(g_diff)

                    causal_mask = torch.tril(
                        torch.ones(cur_BT, cur_BT, device=k.device), diagonal=-1
                    )
                    A_chunk = A_chunk * causal_mask

                    A[b, bos + chunk_start : bos + chunk_end, h, :cur_BT] = A_chunk

    return A


if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, H, K = 2, 32, 4, 16
    BT = 16
    chunk_size = BT

    k = torch.randn(B, T, H, K)
    beta = torch.rand(B, T, H)
    g = torch.rand(B, T, H)

    print(f"Input shapes:")
    print(f"  k: {k.shape}")
    print(f"  beta: {beta.shape}")
    print(f"  g: {g.shape}")

    A = chunk_scaled_dot_kkt_cpu(k, g=g, beta=beta, chunk_size=chunk_size)
    print(f"Output shape: {A.shape}")
    print(f"Output sample (batch 0, head 0):\n{A[0, :, 0, :]}")
