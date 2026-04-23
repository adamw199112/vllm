import torch
import unittest
import sys
import warnings
import math

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/adam.wang/work/vllm")

try:
    from vllm.model_executor.layers.fla.ops.wy_fast import (
        recompute_w_u_fwd as recompute_w_u_triton,
    )

    HAS_TRITON = True
except Exception as e:
    HAS_TRITON = False
    print(f"Triton not available: {e}")


def recompute_w_u_cpu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation of recompute_w_u_fwd kernel (fp16).
    """
    B, T, Hg, K = k.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = A.shape[-1]

    u = k.new_empty(B, T, H, V, dtype=k.dtype)
    w = k.new_empty(B, T, H, K, dtype=k.dtype)

    if cu_seqlens is None:
        NT = math.ceil(T / BT)
    else:
        NT = len(chunk_indices)

    BK = 64
    BV = 64

    for i_t in range(NT):
        chunk_start = i_t * BT
        chunk_end = min(chunk_start + BT, T)
        cur_BT = chunk_end - chunk_start

        for b in range(B):
            for h in range(H):
                k_head_idx = h // (H // Hg)

                beta_slice = beta[b, chunk_start:chunk_end, h]
                A_slice = A[b, chunk_start:chunk_end, h, :cur_BT]
                g_slice = torch.exp(g_cumsum[b, chunk_start:chunk_end, h])

                for i_v in range(math.ceil(V / BV)):
                    v_start = i_v * BV
                    v_end = min(v_start + BV, V)
                    v_slice = v[b, chunk_start:chunk_end, h, v_start:v_end]
                    v_scaled = v_slice * beta_slice[:, None]
                    u_slice = torch.matmul(A_slice[:cur_BT, :cur_BT], v_scaled)
                    u[b, chunk_start:chunk_end, h, v_start:v_end] = u_slice

                for i_k in range(math.ceil(K / BK)):
                    k_start = i_k * BK
                    k_end = min(k_start + BK, K)
                    k_slice = k[b, chunk_start:chunk_end, k_head_idx, k_start:k_end]
                    k_scaled = k_slice * beta_slice[:, None] * g_slice[:, None]
                    w_slice = torch.matmul(A_slice[:cur_BT, :cur_BT], k_scaled)
                    w[b, chunk_start:chunk_end, h, k_start:k_end] = w_slice

    return w, u


def recompute_w_u_cpu_varlen(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for varlen case.
    """
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
            k_head_idx = h // (H // Hg)

            beta_slice = beta[0, bos + chunk_start:bos + chunk_end, h]
            A_slice = A[0, bos + chunk_start:bos + chunk_end, h, :cur_BT]
            g_slice = torch.exp(g_cumsum[0, bos + chunk_start:bos + chunk_end, h])

            for i_v in range(math.ceil(V / BV)):
                v_start = i_v * BV
                v_end = min(v_start + BV, V)
                v_slice = v[0, bos + chunk_start:bos + chunk_end, h, v_start:v_end]
                v_scaled = v_slice * beta_slice[:, None]
                u_slice = torch.matmul(A_slice[:cur_BT, :cur_BT], v_scaled)
                u[0, bos + chunk_start:bos + chunk_end, h, v_start:v_end] = u_slice

            for i_k in range(math.ceil(K / BK)):
                k_start = i_k * BK
                k_end = min(k_start + BK, K)
                k_slice = k[0, bos + chunk_start:bos + chunk_end, k_head_idx, k_start:k_end]
                k_scaled = k_slice * beta_slice[:, None] * g_slice[:, None]
                w_slice = torch.matmul(A_slice[:cur_BT, :cur_BT], k_scaled)
                w[0, bos + chunk_start:bos + chunk_end, h, k_start:k_end] = w_slice

    return w, u


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Local implementation of prepare_chunk_indices for testing."""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


class TestRecomputeWuFwd(unittest.TestCase):
    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_small(self):
        B, T, H, Hg, K, V, BT = 1, 16, 4, 2, 32, 32, 16
        torch.manual_seed(42)

        k = torch.randn(B, T, Hg, K, dtype=torch.float32)
        v = torch.randn(B, T, H, V, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.float32) * 0.5 + 0.5
        g_cumsum = torch.randn(B, T, H, dtype=torch.float32)
        A = torch.randn(B, T, H, BT, dtype=torch.float32)

        w_cpu, u_cpu = recompute_w_u_cpu(k, v, beta, g_cumsum, A)

        w_triton, u_triton = recompute_w_u_triton(
            k.cuda(),
            v.cuda(),
            beta.cuda(),
            g_cumsum.cuda(),
            A.cuda(),
            None,
            None,
        )
        w_triton_cpu = w_triton.cpu()
        u_triton_cpu = u_triton.cpu()

        w_diff = (w_cpu.float() - w_triton_cpu.float()).abs().max().item()
        u_diff = (u_cpu.float() - u_triton_cpu.float()).abs().max().item()
        print(f"Small test - w max diff: {w_diff:.6f}, u max diff: {u_diff:.6f}")
        torch.testing.assert_close(
            w_cpu.float(), w_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            u_cpu.float(), u_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_medium(self):
        B, T, H, Hg, K, V, BT = 2, 64, 8, 4, 64, 64, 32
        torch.manual_seed(123)

        k = torch.randn(B, T, Hg, K, dtype=torch.float32)
        v = torch.randn(B, T, H, V, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.float32) * 0.5 + 0.5
        g_cumsum = torch.randn(B, T, H, dtype=torch.float32)
        A = torch.randn(B, T, H, BT, dtype=torch.float32)

        w_cpu, u_cpu = recompute_w_u_cpu(k, v, beta, g_cumsum, A)

        w_triton, u_triton = recompute_w_u_triton(
            k.cuda(),
            v.cuda(),
            beta.cuda(),
            g_cumsum.cuda(),
            A.cuda(),
            None,
            None,
        )
        w_triton_cpu = w_triton.cpu()
        u_triton_cpu = u_triton.cpu()

        w_diff = (w_cpu.float() - w_triton_cpu.float()).abs().max().item()
        u_diff = (u_cpu.float() - u_triton_cpu.float()).abs().max().item()
        print(f"Medium test - w max diff: {w_diff:.6f}, u max diff: {u_diff:.6f}")
        torch.testing.assert_close(
            w_cpu.float(), w_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            u_cpu.float(), u_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_large(self):
        B, T, H, Hg, K, V, BT = 2, 128, 8, 4, 128, 128, 64
        torch.manual_seed(456)

        k = torch.randn(B, T, Hg, K, dtype=torch.float32)
        v = torch.randn(B, T, H, V, dtype=torch.float32)
        beta = torch.rand(B, T, H, dtype=torch.float32) * 0.5 + 0.5
        g_cumsum = torch.randn(B, T, H, dtype=torch.float32)
        A = torch.randn(B, T, H, BT, dtype=torch.float32)

        w_cpu, u_cpu = recompute_w_u_cpu(k, v, beta, g_cumsum, A)

        w_triton, u_triton = recompute_w_u_triton(
            k.cuda(),
            v.cuda(),
            beta.cuda(),
            g_cumsum.cuda(),
            A.cuda(),
            None,
            None,
        )
        w_triton_cpu = w_triton.cpu()
        u_triton_cpu = u_triton.cpu()

        w_diff = (w_cpu.float() - w_triton_cpu.float()).abs().max().item()
        u_diff = (u_cpu.float() - u_triton_cpu.float()).abs().max().item()
        print(f"Large test - w max diff: {w_diff:.6f}, u max diff: {u_diff:.6f}")
        torch.testing.assert_close(
            w_cpu.float(), w_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            u_cpu.float(), u_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_varlen_16(self):
        print("\n=== Varlen BT=16 ===")
        B = 1
        H = 2
        Hg = 2
        K = 32
        V = 32
        BT = 16
        torch.manual_seed(42)

        lens = torch.tensor([5, 3, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        k = torch.randn(B, total_T, Hg, K, dtype=torch.float32)
        v = torch.randn(B, total_T, H, V, dtype=torch.float32)
        beta = torch.rand(B, total_T, H, dtype=torch.float32) * 0.5 + 0.5
        g_cumsum = torch.randn(B, total_T, H, dtype=torch.float32)
        A = torch.randn(B, total_T, H, BT, dtype=torch.float32)

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        w_cpu, u_cpu = recompute_w_u_cpu_varlen(
            k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices
        )

        w_triton, u_triton = recompute_w_u_triton(
            k.cuda(),
            v.cuda(),
            beta.cuda(),
            g_cumsum.cuda(),
            A.cuda(),
            cu_seqlens.cuda(),
            chunk_indices.cuda(),
        )
        w_triton_cpu = w_triton.cpu()
        u_triton_cpu = u_triton.cpu()

        w_diff = (w_cpu.float() - w_triton_cpu.float()).abs().max().item()
        u_diff = (u_cpu.float() - u_triton_cpu.float()).abs().max().item()
        print(f"Varlen BT=16 - w max diff: {w_diff:.6f}, u max diff: {u_diff:.6f}")
        torch.testing.assert_close(
            w_cpu.float(), w_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            u_cpu.float(), u_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_varlen_32(self):
        print("\n=== Varlen BT=32 ===")
        B = 1
        H = 4
        Hg = 2
        K = 64
        V = 64
        BT = 32
        torch.manual_seed(123)

        lens = torch.tensor([10, 15, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        k = torch.randn(B, total_T, Hg, K, dtype=torch.float32)
        v = torch.randn(B, total_T, H, V, dtype=torch.float32)
        beta = torch.rand(B, total_T, H, dtype=torch.float32) * 0.5 + 0.5
        g_cumsum = torch.randn(B, total_T, H, dtype=torch.float32)
        A = torch.randn(B, total_T, H, BT, dtype=torch.float32)

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        w_cpu, u_cpu = recompute_w_u_cpu_varlen(
            k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices
        )

        w_triton, u_triton = recompute_w_u_triton(
            k.cuda(),
            v.cuda(),
            beta.cuda(),
            g_cumsum.cuda(),
            A.cuda(),
            cu_seqlens.cuda(),
            chunk_indices.cuda(),
        )
        w_triton_cpu = w_triton.cpu()
        u_triton_cpu = u_triton.cpu()

        w_diff = (w_cpu.float() - w_triton_cpu.float()).abs().max().item()
        u_diff = (u_cpu.float() - u_triton_cpu.float()).abs().max().item()
        print(f"Varlen BT=32 - w max diff: {w_diff:.6f}, u max diff: {u_diff:.6f}")
        torch.testing.assert_close(
            w_cpu.float(), w_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            u_cpu.float(), u_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
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

        k = torch.randn(B, total_T, Hg, K, dtype=torch.float32)
        v = torch.randn(B, total_T, H, V, dtype=torch.float32)
        beta = torch.rand(B, total_T, H, dtype=torch.float32) * 0.5 + 0.5
        g_cumsum = torch.randn(B, total_T, H, dtype=torch.float32)
        A = torch.randn(B, total_T, H, BT, dtype=torch.float32)

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        w_cpu, u_cpu = recompute_w_u_cpu_varlen(
            k, v, beta, g_cumsum, A, cu_seqlens, chunk_indices
        )

        w_triton, u_triton = recompute_w_u_triton(
            k.cuda(),
            v.cuda(),
            beta.cuda(),
            g_cumsum.cuda(),
            A.cuda(),
            cu_seqlens.cuda(),
            chunk_indices.cuda(),
        )
        w_triton_cpu = w_triton.cpu()
        u_triton_cpu = u_triton.cpu()

        w_diff = (w_cpu.float() - w_triton_cpu.float()).abs().max().item()
        u_diff = (u_cpu.float() - u_triton_cpu.float()).abs().max().item()
        print(f"Varlen BT=64 - w max diff: {w_diff:.6f}, u max diff: {u_diff:.6f}")
        torch.testing.assert_close(
            w_cpu.float(), w_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            u_cpu.float(), u_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
