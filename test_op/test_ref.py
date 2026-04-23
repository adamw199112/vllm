import torch
import unittest
import sys

sys.path.insert(0, "/home/adam.wang/work/vllm")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from vllm.model_executor.layers.fla.ops.solve_tril import (
        solve_tril as solve_tril_triton,
    )

    HAS_TRITON = True
except Exception as e:
    HAS_TRITON = False
    print(f"Cannot load Triton: {e}")


def solve_tril_ref(A, BT):
    """Using torch.linalg.inv as reference"""
    B, T, H, BT_actual = A.shape
    BT = BT_actual
    Ai = torch.zeros_like(A)

    for b in range(B):
        for h in range(H):
            max_chunks = (T + BT - 1) // BT
            for i_t in range(max_chunks):
                chunk_start = i_t * BT
                chunk_end = min(chunk_start + BT, T)
                cur_BT = chunk_end - chunk_start

                A_chunk = A[b, chunk_start:chunk_end, h, :cur_BT].clone()

                o_i = torch.arange(cur_BT)
                m_I = o_i[:, None] == o_i[None, :]
                m_A = o_i[:, None] > o_i[None, :]

                L = torch.where(m_A, A_chunk, torch.zeros_like(A_chunk))
                M = -L + m_I.float()

                try:
                    M_inv = torch.linalg.inv(M)
                except:
                    M_inv = torch.zeros_like(M)

                Ai[b, chunk_start:chunk_end, h, :cur_BT] = M_inv
    return Ai


class TestSolveTril(unittest.TestCase):
    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_16(self):
        print("\n=== CPU vs Triton BT=16 ===")
        B, T, H = 1, 16, 1
        torch.manual_seed(42)
        A = torch.randn(B, T, H, 16) * 0.1
        mask = torch.tril(torch.ones(T, 16, dtype=torch.bool), diagonal=-1)
        A = A * mask.unsqueeze(0).unsqueeze(2)

        # Reference using torch.linalg.inv
        Ai_ref = solve_tril_ref(A.clone(), 16)

        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(A_gpu)
        Ai_triton_cpu = Ai_triton.cpu()

        diff = (Ai_ref.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff (ref vs Triton): {diff:.6f}")

        print("=== Ref ===")
        print(Ai_ref[0, :, 0, :])
        print("\n=== Triton ===")
        print(Ai_triton_cpu[0, :, 0, :])


if __name__ == "__main__":
    unittest.main(verbosity=2)
