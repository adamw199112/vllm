# Verification using torch.testing.assert_close

import torch
import math
import sys

sys.path.insert(0, "/home/adam.wang/work/vllm")

from chunk_scaled_dot_kkt_cpu import chunk_scaled_dot_kkt_cpu


def test_without_gate():
    """Test without gate."""
    print("Test 1: Without gate")
    torch.manual_seed(42)
    B, T, H, K = 2, 32, 4, 16
    BT = 16

    k = torch.randn(B, T, H, K)
    beta = torch.rand(B, T, H)

    k_gpu = k.cuda()
    beta_gpu = beta.cuda()

    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )

    A_gpu = chunk_scaled_dot_kkt_fwd(k_gpu, beta=beta_gpu, chunk_size=BT)
    A_cpu = chunk_scaled_dot_kkt_cpu(k, beta=beta, chunk_size=BT)

    try:
        torch.testing.assert_close(A_gpu.cpu(), A_cpu, atol=1e-2, rtol=1e-2)
        print("  PASSED!")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_with_gate():
    """Test with gate g."""
    print("Test 2: With gate")
    torch.manual_seed(42)
    B, T, H, K = 2, 32, 4, 16
    BT = 16

    k = torch.randn(B, T, H, K)
    beta = torch.rand(B, T, H)
    g = torch.rand(B, T, H)

    k_gpu = k.cuda()
    beta_gpu = beta.cuda()
    g_gpu = g.cuda()

    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )

    A_gpu = chunk_scaled_dot_kkt_fwd(k_gpu, g=g_gpu, beta=beta_gpu, chunk_size=BT)
    A_cpu = chunk_scaled_dot_kkt_cpu(k, g=g, beta=beta, chunk_size=BT)

    try:
        torch.testing.assert_close(A_gpu.cpu(), A_cpu, atol=1e-2, rtol=1e-2)
        print("  PASSED!")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_hg_neq_h():
    """Test with Hg != H."""
    print("Test 3: Hg != H")
    torch.manual_seed(42)
    B, T, H, K = 1, 32, 8, 16
    Hg = 4
    BT = 16

    k = torch.randn(B, T, Hg, K)
    beta = torch.rand(B, T, H)

    k_gpu = k.cuda()
    beta_gpu = beta.cuda()

    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )

    A_gpu = chunk_scaled_dot_kkt_fwd(k_gpu, beta=beta_gpu, chunk_size=BT)

    A_cpu = chunk_scaled_dot_kkt_cpu(k, beta=beta, chunk_size=BT)

    try:
        torch.testing.assert_close(A_gpu.cpu(), A_cpu, atol=1e-2, rtol=1e-2)
        print("  PASSED!")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_varlen():
    """Test with variable length sequences."""
    print("Test 4: Varlen without gate")
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    B = 1
    H = 4
    K = 16
    BT = 16

    lens = torch.tensor([5, 3, 7], dtype=torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
    total_tokens = cu_seqlens[-1].item()

    k_cpu = torch.randn(B, total_tokens, H, K)
    beta_cpu = torch.rand(B, total_tokens, H)

    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    A_cpu = chunk_scaled_dot_kkt_cpu(
        k_cpu,
        beta=beta_cpu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )

    k_gpu = k_cpu.cuda()
    beta_gpu = beta_cpu.cuda()
    cu_seqlens_gpu = cu_seqlens.cuda()
    chunk_indices_gpu = chunk_indices.cuda()

    A_gpu = chunk_scaled_dot_kkt_fwd(
        k_gpu,
        beta=beta_gpu,
        cu_seqlens=cu_seqlens_gpu,
        chunk_indices=chunk_indices_gpu,
        chunk_size=BT,
    )

    try:
        torch.testing.assert_close(A_gpu.cpu(), A_cpu, atol=1e-2, rtol=1e-2)
        print("  PASSED!")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_varlen_with_gate():
    """Test with variable length sequences and gate."""
    print("Test 5: Varlen with gate")
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    B = 1
    H = 4
    K = 16
    BT = 16

    lens = torch.tensor([5, 3, 7], dtype=torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
    total_tokens = cu_seqlens[-1].item()

    k_cpu = torch.randn(B, total_tokens, H, K)
    beta_cpu = torch.rand(B, total_tokens, H)
    g_cpu = torch.rand(B, total_tokens, H)

    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    A_cpu = chunk_scaled_dot_kkt_cpu(
        k_cpu,
        g=g_cpu,
        beta=beta_cpu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )

    k_gpu = k_cpu.cuda()
    beta_gpu = beta_cpu.cuda()
    g_gpu = g_cpu.cuda()
    cu_seqlens_gpu = cu_seqlens.cuda()
    chunk_indices_gpu = chunk_indices.cuda()

    A_gpu = chunk_scaled_dot_kkt_fwd(
        k_gpu,
        g=g_gpu,
        beta=beta_gpu,
        cu_seqlens=cu_seqlens_gpu,
        chunk_indices=chunk_indices_gpu,
        chunk_size=BT,
    )

    try:
        torch.testing.assert_close(A_gpu.cpu(), A_cpu, atol=1e-2, rtol=1e-2)
        print("  PASSED!")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_varlen_two_sequences():
    """Test with two sequences in varlen mode."""
    print("Test 6: Varlen two sequences")
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    lens = torch.tensor([5, 7], dtype=torch.int32)
    cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])
    total_tokens = cu_seqlens[-1].item()

    B = 1
    H = 4
    K = 16
    BT = 16

    k_cpu = torch.randn(B, total_tokens, H, K)
    beta_cpu = torch.rand(B, total_tokens, H)

    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    A_cpu = chunk_scaled_dot_kkt_cpu(
        k_cpu,
        beta=beta_cpu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )

    k_gpu = k_cpu.cuda()
    beta_gpu = beta_cpu.cuda()
    cu_seqlens_gpu = cu_seqlens.cuda()
    chunk_indices_gpu = chunk_indices.cuda()

    A_gpu = chunk_scaled_dot_kkt_fwd(
        k_gpu,
        beta=beta_gpu,
        cu_seqlens=cu_seqlens_gpu,
        chunk_indices=chunk_indices_gpu,
        chunk_size=BT,
    )

    try:
        torch.testing.assert_close(A_gpu.cpu(), A_cpu, atol=1e-2, rtol=1e-2)
        print("  PASSED!")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    results = []

    results.append(test_without_gate())
    results.append(test_with_gate())
    results.append(test_hg_neq_h())
    results.append(test_varlen())
    results.append(test_varlen_with_gate())
    results.append(test_varlen_two_sequences())

    print("=" * 60)
    print(f"Total: {sum(results)}/{len(results)} passed")
