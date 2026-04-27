import torch
import math
import sys
import warnings

warnings.filterwarnings("ignore")


def exp(x):
    return torch.exp(x)


def chunk_fwd_o_cpu(
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
    B, T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        N = B
        NT = math.ceil(T / BT)
    else:
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)

    if scale is None:
        scale = K ** -0.5

    o = torch.empty_like(v)

    for b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            cur_BT = t_end - t_start

            if cur_BT == 0:
                continue

            for h_idx in range(H):
                k_head_idx = h_idx // (H // Hg) if H != Hg else h_idx

                q_chunk = q[b, t_start:t_end, k_head_idx, :]
                k_chunk = k[b, t_start:t_end, k_head_idx, :]
                v_chunk = v[b, t_start:t_end, h_idx, :]
                h_state = h[b, i_t, h_idx]

                o_chunk = torch.matmul(q_chunk, h_state.T)

                A_chunk = torch.matmul(q_chunk, k_chunk.T)

                if g is not None:
                    g_chunk = g[b, t_start:t_end, h_idx]
                    o_chunk = o_chunk * exp(g_chunk[:, None])
                    A_chunk = A_chunk * exp(g_chunk[:, None] - g_chunk[None, :])

                mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=torch.bool, device=q.device))
                A_chunk = A_chunk.masked_fill(~mask, 0)

                o_chunk = o_chunk + torch.matmul(A_chunk, v_chunk)
                o_chunk = o_chunk * scale

                o[b, t_start:t_end, h_idx, :] = o_chunk

    return o


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

            q_chunk = q[0, bos + t_start:bos + t_end, k_head_idx, :]
            k_chunk = k[0, bos + t_start:bos + t_end, k_head_idx, :]
            v_chunk = v[0, bos + t_start:bos + t_end, h_idx, :]
            h_state = h[idx, h_idx]

            o_chunk = torch.matmul(q_chunk, h_state.T)

            A_chunk = torch.matmul(q_chunk, k_chunk.T)

            if g is not None:
                g_chunk = g[0, bos + t_start:bos + t_end, h_idx]
                o_chunk = o_chunk * exp(g_chunk[:, None])
                A_chunk = A_chunk * exp(g_chunk[:, None] - g_chunk[None, :])

            mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=torch.bool, device=q.device))
            A_chunk = A_chunk.masked_fill(~mask, 0)

            o_chunk = o_chunk + torch.matmul(A_chunk, v_chunk)
            o_chunk = o_chunk * scale

            o[0, bos + t_start:bos + t_end, h_idx, :] = o_chunk

    return o


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


sys.path.insert(0, "/home/adam.wang/work/vllm/test_op/op_release")
from chunk_o import chunk_fwd_o as chunk_fwd_o_triton


import unittest


class TestChunkO(unittest.TestCase):
    def test_basic(self):
        print("\n=== Basic ===")
        B, T, Hg, K, H, V = 1, 32, 1, 16, 1, 8
        BT = 32
        torch.manual_seed(42)
        q = torch.randn(B, T, Hg, K, dtype=torch.float16) * 0.1
        k = torch.randn(B, T, Hg, K, dtype=torch.float16) * 0.1
        v = torch.randn(B, T, H, V, dtype=torch.float16) * 0.1
        h = torch.randn(B, math.ceil(T/BT), H, V, K, dtype=torch.float16) * 0.1

        o_cpu = chunk_fwd_o_cpu(q, k, v, h, chunk_size=BT)
        o_triton = chunk_fwd_o_triton(
            q=q.cuda(), k=k.cuda(), v=v.cuda(), h=h.cuda(), chunk_size=BT,
        )

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"Basic - o max diff: {o_diff:.6f}")
        self.assertTrue(o_diff < 1e-1)

    def test_basic_with_gate(self):
        print("\n=== Basic with gate ===")
        B, T, Hg, K, H, V = 1, 32, 1, 16, 1, 8
        BT = 32
        torch.manual_seed(42)
        q = torch.randn(B, T, Hg, K, dtype=torch.float16) * 0.1
        k = torch.randn(B, T, Hg, K, dtype=torch.float16) * 0.1
        v = torch.randn(B, T, H, V, dtype=torch.float16) * 0.1
        h = torch.randn(B, math.ceil(T/BT), H, V, K, dtype=torch.float16) * 0.1
        g = torch.randn(B, T, H, dtype=torch.float16) * 0.1

        o_cpu = chunk_fwd_o_cpu(q, k, v, h, g=g, chunk_size=BT)
        o_triton = chunk_fwd_o_triton(
            q=q.cuda(), k=k.cuda(), v=v.cuda(), h=h.cuda(), g=g.cuda(), chunk_size=BT,
        )

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"Basic with gate - o max diff: {o_diff:.6f}")
        self.assertTrue(o_diff < 1e-1)

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

        q = torch.randn(B, total_T, Hg, K, dtype=torch.float16) * 0.1
        k = torch.randn(B, total_T, Hg, K, dtype=torch.float16) * 0.1
        v = torch.randn(B, total_T, H, V, dtype=torch.float16) * 0.1
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        num_chunks = len(chunk_indices)
        h = torch.randn(num_chunks, H, V, K, dtype=torch.float16) * 0.1

        o_cpu = chunk_fwd_o_cpu_varlen(
            q, k, v, h, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        o_triton = chunk_fwd_o_triton(
            q=q.cuda(), k=k.cuda(), v=v.cuda(), h=h.cuda(), chunk_size=BT,
            cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda(),
        )

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"Varlen BT=16 - o max diff: {o_diff:.6f}")
        self.assertTrue(o_diff < 1e-1)

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

        q = torch.randn(B, total_T, Hg, K, dtype=torch.float16) * 0.1
        k = torch.randn(B, total_T, Hg, K, dtype=torch.float16) * 0.1
        v = torch.randn(B, total_T, H, V, dtype=torch.float16) * 0.1
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        num_chunks = len(chunk_indices)
        h = torch.randn(num_chunks, H, V, K, dtype=torch.float16) * 0.1

        o_cpu = chunk_fwd_o_cpu_varlen(
            q, k, v, h, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        o_triton = chunk_fwd_o_triton(
            q=q.cuda(), k=k.cuda(), v=v.cuda(), h=h.cuda(), chunk_size=BT,
            cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda(),
        )

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"Varlen BT=32 - o max diff: {o_diff:.6f}")
        self.assertTrue(o_diff < 1e-1)

    def test_varlen_64(self):
        print("\n=== Varlen BT=64 ===")
        B = 1
        H = 4
        Hg = 2
        K = 64
        V = 32
        BT = 64
        torch.manual_seed(456)

        lens = torch.tensor([20, 15, 33], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
        total_T = cu_seqlens[-1].item()

        q = torch.randn(B, total_T, Hg, K, dtype=torch.float16) * 0.1
        k = torch.randn(B, total_T, Hg, K, dtype=torch.float16) * 0.1
        v = torch.randn(B, total_T, H, V, dtype=torch.float16) * 0.1
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        num_chunks = len(chunk_indices)
        h = torch.randn(num_chunks, H, V, K, dtype=torch.float16) * 0.1

        o_cpu = chunk_fwd_o_cpu_varlen(
            q, k, v, h, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        o_triton = chunk_fwd_o_triton(
            q=q.cuda(), k=k.cuda(), v=v.cuda(), h=h.cuda(), chunk_size=BT,
            cu_seqlens=cu_seqlens.cuda(), chunk_indices=chunk_indices.cuda(),
        )

        o_diff = (o_cpu.float() - o_triton.cpu().float()).abs().max().item()
        print(f"Varlen BT=64 - o max diff: {o_diff:.6f}")
        self.assertTrue(o_diff < 1e-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)