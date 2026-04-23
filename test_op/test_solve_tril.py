import torch
import unittest
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/adam.wang/work/vllm")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

try:
    from vllm.model_executor.layers.fla.ops.solve_tril import (
        solve_tril as solve_tril_triton,
    )

    HAS_TRITON = True
    print("Triton module loaded successfully")
except Exception as e:
    HAS_TRITON = False
    print(f"Cannot load Triton: {e}")


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Local implementation of prepare_chunk_indices for testing."""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def solve_tril_cpu(A: torch.Tensor) -> torch.Tensor:
    """CPU implementation using forward substitution (same as Triton)."""
    B, T, H, BT = A.shape
    Ai = torch.zeros_like(A)

    for b in range(B):
        for h in range(H):
            max_chunks = (T + BT - 1) // BT
            for i_t in range(max_chunks):
                chunk_start = i_t * BT
                chunk_end = min(chunk_start + BT, T)
                cur_BT = chunk_end - chunk_start

                A_chunk = A[b, chunk_start:chunk_end, h, :cur_BT].clone()

                m_A = torch.tril(
                    torch.ones(cur_BT, cur_BT, dtype=torch.bool), diagonal=-1
                )
                A_masked = torch.where(
                    m_A, A_chunk, torch.zeros(cur_BT, cur_BT, dtype=A.dtype)
                )
                I = torch.eye(cur_BT, cur_BT, dtype=A.dtype)
                M = I + A_masked
                try:
                    M_inv = torch.linalg.inv(M.to(torch.float32)).to(A.dtype)
                except RuntimeError:
                    M_inv = torch.zeros_like(M)

                Ai[b, chunk_start:chunk_end, h, :cur_BT] = M_inv
    return Ai


def solve_tril_cpu_varlen(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    BT: int,
) -> torch.Tensor:
    """CPU implementation for varlen case using column-wise forward substitution."""
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


class TestSolveTril(unittest.TestCase):
    def _verify_inverse(self, A, Ai, atol=1e-3, rtol=1e-3):
        B, T, H, BT = A.shape
        for b in range(B):
            for h in range(H):
                for i_t in range((T + BT - 1) // BT):
                    chunk_start = i_t * BT
                    chunk_end = min(chunk_start + BT, T)
                    cur_BT = chunk_end - chunk_start
                    if cur_BT == 0:
                        continue
                    A_tril = torch.tril(A[b, chunk_start:chunk_end, h, :cur_BT].clone())
                    I = torch.eye(cur_BT, cur_BT, dtype=A.dtype)
                    M = I + A_tril
                    M_inv = Ai[b, chunk_start:chunk_end, h, :cur_BT]
                    result = M_inv @ M
                    torch.testing.assert_close(result, I, atol=atol, rtol=rtol)

    def test_cpu_vs_triton_16(self):
        print("\n=== CPU vs Triton BT=16 ===")
        B, T, H = 1, 16, 1
        torch.manual_seed(42)
        A = torch.randn(B, T, H, 16, dtype=torch.float16) * 0.1
        mask = torch.tril(torch.ones(T, 16, dtype=torch.bool), diagonal=-1)
        A = A * mask.unsqueeze(0).unsqueeze(2)

        Ai_cpu = solve_tril_cpu(A.clone())

        A_cuda = A.clone().cuda()
        Ai_triton = solve_tril_triton(A_cuda)
        Ai_triton_cpu = Ai_triton.cpu()

        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        torch.testing.assert_close(
            Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        print("CPU vs Triton BT=16: PASSED")

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_32(self):
        print("\n=== CPU vs Triton BT=32 ===")
        B, T, H = 1, 32, 1
        torch.manual_seed(42)
        A = torch.randn(B, T, H, 32, dtype=torch.float16) * 0.1
        mask = torch.tril(torch.ones(T, 32, dtype=torch.bool), diagonal=-1)
        A = A * mask.unsqueeze(0).unsqueeze(2)

        Ai_cpu = solve_tril_cpu(A.clone())

        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(A_gpu)
        Ai_triton_cpu = Ai_triton.cpu()

        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        torch.testing.assert_close(
            Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        print("CPU vs Triton BT=32: PASSED")

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_64(self):
        print("\n=== CPU vs Triton BT=64 ===")
        B, T, H = 1, 64, 1
        torch.manual_seed(42)
        A = torch.randn(B, T, H, 64, dtype=torch.float16) * 0.1
        mask = torch.tril(torch.ones(T, 64, dtype=torch.bool), diagonal=-1)
        A = A * mask.unsqueeze(0).unsqueeze(2)
        

        Ai_cpu = solve_tril_cpu(A.clone())

        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(A_gpu)
        Ai_triton_cpu = Ai_triton.cpu()

        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")

        torch.testing.assert_close(
            Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        print("CPU vs Triton BT=64: PASSED")

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_varlen_16(self):
        print("\n=== Varlen BT=16 ===")
        B = 1
        H = 2
        BT = 16
        torch.manual_seed(42)

        lens = torch.tensor([5, 3, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        A = torch.randn(B, total_T, H, BT, dtype=torch.float16) * 0.1
        for i in range(len(lens)):
            bos = int(cu_seqlens[i].item())
            eos = int(cu_seqlens[i + 1].item())
            T_seq = eos - bos
            seq_mask = torch.tril(torch.ones(T_seq, BT, dtype=torch.bool), diagonal=-1)
            A[0, bos:eos] = A[0, bos:eos] * seq_mask.unsqueeze(1)

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        Ai_cpu = solve_tril_cpu_varlen(A.clone(), cu_seqlens, chunk_indices, BT)

        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(
            A_gpu, cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda()
        )
        Ai_triton_cpu = Ai_triton.cpu()

        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        torch.testing.assert_close(
            Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        print("Varlen BT=16: PASSED")

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_varlen_32(self):
        print("\n=== Varlen BT=32 ===")
        B = 1
        H = 2
        BT = 32
        torch.manual_seed(42)

        lens = torch.tensor([10, 15, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        A = torch.randn(B, total_T, H, BT, dtype=torch.float16) * 0.1
        for i in range(len(lens)):
            bos = int(cu_seqlens[i].item())
            eos = int(cu_seqlens[i + 1].item())
            T_seq = eos - bos
            seq_mask = torch.tril(torch.ones(T_seq, BT, dtype=torch.bool), diagonal=-1)
            A[0, bos:eos] = A[0, bos:eos] * seq_mask.unsqueeze(1)

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        Ai_cpu = solve_tril_cpu_varlen(A.clone(), cu_seqlens, chunk_indices, BT)

        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(
            A_gpu, cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda()
        )
        Ai_triton_cpu = Ai_triton.cpu()

        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        torch.testing.assert_close(
            Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        print("Varlen BT=32: PASSED")

    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_varlen_64(self):
        print("\n=== Varlen BT=64 ===")
        B = 1
        H = 2
        BT = 64
        torch.manual_seed(42)

        lens = torch.tensor([20, 15, 33], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        A = torch.randn(B, total_T, H, BT, dtype=torch.float16) * 0.1
        for i in range(len(lens)):
            bos = int(cu_seqlens[i].item())
            eos = int(cu_seqlens[i + 1].item())
            T_seq = eos - bos
            seq_mask = torch.tril(torch.ones(T_seq, BT, dtype=torch.bool), diagonal=-1)
            A[0, bos:eos] = A[0, bos:eos] * seq_mask.unsqueeze(1)

        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        Ai_cpu = solve_tril_cpu_varlen(A.clone(), cu_seqlens, chunk_indices, BT)

        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(
            A_gpu, cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda()
        )
        Ai_triton_cpu = Ai_triton.cpu()

        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        torch.testing.assert_close(
            Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-3, rtol=1e-3
        )
        print("Varlen BT=64: PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
