# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import numpy as np
import sys
sys.path.append("../")
from   vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
import torch
import math


def chunk_local_cumsum_cpu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = None,
):
    """
    CPU reference of chunk_local_cumsum_kernel
    完全等价 Triton 行为（chunk 内 scan，不做跨 chunk 修正）
    """

    assert chunk_size & (chunk_size - 1) == 0, "chunk_size must be power of 2"
    
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape


    if output_dtype is None:
        output_dtype = g.dtype

    # 输出 tensor
    o = torch.empty_like(g, dtype=output_dtype)

    # ===== case 1: 非 varlen =====
    if cu_seqlens is None:
        for b in range(B):
            for h in range(H):
                # 取一条序列
                if head_first:
                    seq = g[b, h]  # [T]
                else:
                    seq = g[b, :, h]

                out_seq = torch.empty_like(seq, dtype=output_dtype)

                # 分 chunk
                for t0 in range(0, T, chunk_size):
                    t1 = min(t0 + chunk_size, T)
                    chunk = seq[t0:t1].to(torch.float32)

                    # local cumsum
                    cumsum = torch.cumsum(chunk, dim=0)

                    if reverse:
                        S = chunk.sum()
                        cumsum = -cumsum + S + chunk

                    out_seq[t0:t1] = cumsum.to(output_dtype)

                # 写回
                if head_first:
                    o[b, h] = out_seq
                else:
                    o[b, :, h] = out_seq

        return o

    # ===== case 2: varlen =====
    else:
        assert chunk_indices is not None

        # 扁平视角访问（和 Triton 一致）
        # g: [total_tokens, H] 或等价 reshape
        if head_first:
            raise NotImplementedError("varlen + head_first=True 不常见，这里可扩展")

        total_tokens = g.shape[0]

        # 遍历每个 chunk（对应 Triton 的 program_id(0)）
        for idx in range(len(chunk_indices)):
            i_n = int(chunk_indices[idx, 0].item())  # 序列 id
            i_t = int(chunk_indices[idx, 1].item())  # chunk id

            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            T_seq = eos - bos

            # 当前 chunk 在该序列中的范围
            t0 = i_t * chunk_size
            t1 = min(t0 + chunk_size, T_seq)

            if t0 >= T_seq:
                continue

            # 遍历 head
            for h in range(H):
                # global offset
                start = bos + t0
                end = bos + t1

                chunk = g[start:end, h].to(torch.float32)

                cumsum = torch.cumsum(chunk, dim=0)

                if reverse:
                    S = chunk.sum()
                    cumsum = -cumsum + S + chunk

                o[start:end, h] = cumsum.to(output_dtype)

        return o

class TestChunkLocalCumsumCPU:
    
    def test_scalar_head_first_no_reverse(self):
        
       
        
        B, H, T = 2, 4, 16
        chunk_size = 4
        g = torch.randn(B, H, T).cuda()
        
        result_gpu = chunk_local_cumsum(g, chunk_size, reverse=False, head_first=True)
        result_cpu = chunk_local_cumsum_cpu(g, chunk_size, reverse=False, head_first=True)
        
        torch.testing.assert_close(result_cpu.float(), result_gpu.float(), atol=1e-3, rtol=1e-3)
    
    def test_scalar_head_first_reverse(self):
 
        
        B, H, T = 2, 4, 16
        chunk_size = 4
        g = torch.randn(B, H, T).cuda()
        
        result_gpu = chunk_local_cumsum(g, chunk_size, reverse=True, head_first=True)
        result_cpu = chunk_local_cumsum_cpu(g, chunk_size, reverse=True, head_first=True)
        
        torch.testing.assert_close(result_cpu.float(), result_gpu.float(), atol=1e-3, rtol=1e-3)
    
    def test_scalar_no_head_first_no_reverse(self):

        
        B, T, H = 2, 16, 4
        chunk_size = 4
        g = torch.randn(B, T, H).cuda()
        
        result_gpu = chunk_local_cumsum(g, chunk_size, reverse=False, head_first=False)
        result_cpu = chunk_local_cumsum_cpu(g, chunk_size, reverse=False, head_first=False)
        
        torch.testing.assert_close(result_cpu.float(), result_gpu.float(), atol=1e-3, rtol=1e-3)
    
    def test_scalar_no_head_first_reverse(self):
        

        
        B, T, H = 2, 16, 4
        chunk_size = 4
        g = torch.randn(B, T, H).cuda()
        
        result_gpu = chunk_local_cumsum(g, chunk_size, reverse=True, head_first=False)
        result_cpu = chunk_local_cumsum_cpu(g, chunk_size, reverse=True, head_first=False)
        
        torch.testing.assert_close(result_cpu.float(), result_gpu.float(), atol=1e-3, rtol=1e-3)
    
    
    def test_chunk_local_cumsum_varlen(self):
        from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices
        torch.manual_seed(0)

        H = 2
        B = 1
        chunk_size = 4

        # ===== 构造变长序列 =====
        # seq lengths: [5, 3, 7]
        lens = torch.tensor([5, 3, 7], dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0]), lens.cumsum(0)])

        total_tokens = cu_seqlens[-1].item()

        # packed tensor: [total_tokens, H]
        g = torch.randn(B,total_tokens, H, dtype=torch.float32)

        # chunk mapping
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        # ===== CPU =====
        out_cpu = chunk_local_cumsum_cpu(
            g,
            chunk_size,
            reverse=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            head_first=False,
        )

        # ===== Triton =====
        out_triton = chunk_local_cumsum(
            g.cuda(),
            chunk_size,
            reverse=False,
            cu_seqlens=cu_seqlens.cuda(),
            chunk_indices=chunk_indices.cuda(),
            head_first=False,
        )

        print("Varlen forward match:", torch.allclose(out_cpu, out_triton.cpu(), atol=1e-5))

        # ===== reverse =====
        out_cpu_rev = chunk_local_cumsum_cpu(
            g,
            chunk_size,
            reverse=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            head_first=False,
        )

        out_triton_rev = chunk_local_cumsum(
            g.cuda(),
            chunk_size,
            reverse=True,
            cu_seqlens=cu_seqlens.cuda(),
            chunk_indices=chunk_indices.cuda(),
            head_first=False,
        )

        print("Varlen reverse match:", torch.allclose(out_cpu_rev, out_triton_rev.cpu(), atol=1e-5))





if __name__ == "__main__":
    pytest.main([__file__, "-v"])