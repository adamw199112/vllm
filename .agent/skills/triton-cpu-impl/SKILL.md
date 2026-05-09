---
name: triton-cpu-impl
description: "Convert between Triton GPU kernels and CPU implementations. Use this when: (1) writing CPU implementation from a Triton kernel for testing/debugging, (2) writing Triton kernel from CPU implementation for GPU acceleration. Trigger when user mentions 'CPU implementation', 'Triton kernel', 'write triton from CPU', 'write CPU from triton', or similar."
---

# Triton ↔ CPU Implementation Conversion

Bidirectional conversion between Triton GPU kernels and pure PyTorch CPU implementations.

---

## Quick Reference

| Direction | When to Use |
|-----------|------------|
| Triton → CPU | Debug/test GPU kernel, CPU-only environment |
| CPU → Triton | Accelerate CPU code on GPU |

---

## Part 1: Triton → CPU Implementation

### When to Use

- Debug GPU kernel behavior
- Create reference implementation for testing
- Run in CPU-only environment
- Verify correctness against GPU output

### Pattern Analysis

**1. Identify kernel computation pattern:**

```python
# Triton kernel structure:
@triton.jit
def kernel_name(
    k, g, beta, A,  # input tensors
    cu_seqlens, chunk_indices,
    T, H, K, BT, BK,  # shapes
    IS_VARLEN, USE_G,  # flags
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    # Load data (block_ptr pattern):
    p_k = tl.make_block_ptr(k + ..., (T, K), (Hg*K, 1), (i_t*BT, i_k*BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    
    # Computation:
    b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
    
    # Mask (causal):
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    
    # Store:
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
```

**2. Extract key information:**
- Input tensor shapes
- Output computation formula
- Mask/condition logic
- Iteration pattern (grid size)

### Conversion Steps

1. **Map tensor shapes:**
   - `[B, T, H, K]` → CPU indexing by batch, time, head, key

2. **Replace Triton ops with PyTorch:**
   - `tl.make_block_ptr` → array slicing
   - `tl.load` → tensor indexing
   - `tl.dot` → `torch.matmul`
   - `tl.trans` → `.T` or `.transpose(-2, -1)`
   - `tl.where` → `torch.where` or masking
   - `tl.exp` → `torch.exp`

3. **Handle causal mask:**
   - `o_t[:, None] > o_t[None, :]` → `torch.tril(..., diagonal=-1)` (lower tri)
   - `o_t[:, None] >= o_t[None, :]` → `torch.tril(..., diagonal=0)` (with diagonal)
   - Strict upper: `torch.triu(..., diagonal=1)` (exclude diagonal)

4. **Map grid to loops:**
   - `NT, B*H` grid → `for i_t in range(NT): for b in range(B): for h in range(H):`

### Example Transformation

**Original Triton:**
```python
@triton.jit
def chunk_scaled_dot_kkt_fwd_kernel(
    k, beta, g, A, T, H, K, BT, BK, USE_G
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + bos*H + i_h, (T,K), (Hg*K,1), (i_t*BT, i_k*BK), (BT,BK), (1,0))
        b_k = tl.load(p_k, boundary_check=(0,1))
        b_A += tl.dot((b_k * b_beta[:, None]).to(b_k.dtype), tl.trans(b_k))
    
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    tl.store(p_A, b_A)
```

**CPU Implementation:**
```python
import torch
import math

def chunk_scaled_dot_kkt_cpu(k, beta, g=None, chunk_size=64):
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    
    A = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    
    for i_t in range(math.ceil(T / BT)):
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
                
                # Causal mask: lower triangular
                causal_mask = torch.tril(torch.ones(cur_BT, cur_BT), diagonal=-1)
                A_chunk = A_chunk * causal_mask
                
                A[b, chunk_start:chunk_end, h, :cur_BT] = A_chunk
    
    return A
```

### Common Triton → PyTorch Mappings

| Triton Operation | PyTorch Equivalent |
|------------------|-------------------|
| `tl.load(ptr)` | `tensor[idx]` |
| `tl.store(ptr, val)` | `out[idx] = val` |
| `tl.dot(a, b)` | `torch.matmul(a, b)` |
| `tl.trans(x)` | `x.T` or `x.transpose(-2, -1)` |
| `tl.exp(x)` | `torch.exp(x)` |
| `tl.sum(x, axis)` | `x.sum(dim=axis)` |
| `tl.max(x, axis)` | `x.max(dim=axis)[0]` |
| `tl.arange(0, n)` | `torch.arange(n)` |
| `tl.make_block_ptr(...)` | Array slicing |
| `tl.where(mask, a, b)` | `torch.where(mask, a, b)` |
| `o_t[:, None] > o_t[None, :]` | `torch.tril(ones)` |

---

## Part 2: CPU Implementation → Triton Kernel

### When to Use

- Accelerate CPU implementation on GPU
- Optimize for production use

### Pattern Analysis

**1. Analyze CPU implementation:**
```python
def cpu_impl(k, beta, g=None):
    B, T, H, K = k.shape
    
    for i_t in range(ceil(T / BT)):
        for b in range(B):
            for h in range(H):
                beta_slice = beta[b, t_start:t_end, h]
                k_slice = k[b, t_start:t_end, h, :]
                
                k_beta = k_slice * beta_slice[:, None]
                A_chunk = k_beta @ k_slice.T
                
                if g is not None:
                    g_diff = g[:, None] - g[None, :]
                    A_chunk = A_chunk * exp(g_diff)
                
                causal = tril(ones)
                A_chunk = A_chunk * causal
```

### Conversion Steps

1. **Map loops to grid:**
   - `for i_t in range(NT)` → `tl.program_id(0)`
   - `for b in range(B): for h in range(H)` → `i_bh = tl.program_id(1)`, `i_b = i_bh // H`, `i_h = i_bh % H`

2. **Replace PyTorch with Triton:**
   - Indexing → `tl.make_block_ptr` + `tl.load`/`tl.store`
   - `torch.matmul` → `tl.dot`
   - `.T` → `tl.trans`

3. **Add mask logic:**
   - `m_t = o_t < T` for boundary check
   - `m_A = (o_t[:, None] > o_t[None, :]) & m_t` for causal

4. **Write kernel:**
```python
@triton.jit
def chunk_scaled_dot_kkt_kernel(
    k, beta, g, A,
    T, H, K, BT, BK,
    USE_G: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + bos * H + i_h // (H // Hg),
            (T, K), (Hg * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot((b_k * b_beta[:, None]).to(b_k.dtype), tl.trans(b_k))
    
    if USE_G:
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)
    
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
```

---

## Part 3: Verification (MANDATORY)

### Critical Requirement

**Verification is MANDATORY for all conversions.** You MUST import the existing implementation and compare results.

### Test Structure for Triton → CPU

```python
import torch
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "/home/adam.wang/work/vllm")

# Import EXTSTING TRITON implementation
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril as solve_tril_triton

# Your CPU implementation
def solve_tril_cpu(A: torch.Tensor) -> torch.Tensor:
    # ... your implementation
    pass

class TestSolveTril(unittest.TestCase):
    
    def _verify_inverse(self, A, Ai, atol=1e-3, rtol=1e-3):
        # Verify (I + A)^-1 * (I + A) = I
        ...
    
    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_16(self):
        B, T, H = 1, 16, 1
        torch.manual_seed(42)
        A = torch.randn(B, T, H, 16) * 0.1
        A = torch.tril(A, diagonal=-1)
        
        # Run your CPU implementation
        Ai_cpu = solve_tril_cpu(A.clone())
        
        # Run EXISTING TRITON implementation
        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(A_gpu)
        Ai_triton_cpu = Ai_triton.cpu()
        
        # Compare
        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        torch.testing.assert_close(Ai_cpu.float(), Ai_triton_cpu.float(), atol=atol, rtol=rtol)
```

### Test Structure for CPU → Triton

```python
import torch
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "/home/adam.wang/work/vllm")

# Import EXISTING CPU implementation (reference)
from your_cpu_module import cpu_func as cpu_reference

# Your Triton implementation
def triton_func(A: torch.Tensor) -> torch.Tensor:
    # ... your Triton implementation
    pass

class TestCompare(unittest.TestCase):
    
    def test_triton_vs_cpu_reference(self):
        torch.manual_seed(42)
        A = torch.randn(B, T, H, K) * 0.1
        
        # Run existing CPU reference
        Ai_cpu = cpu_reference(A.clone())
        
        # Run your Triton implementation
        A_gpu = A.clone().cuda()
        Ai_triton = triton_func(A_gpu)
        Ai_triton_cpu = Ai_triton.cpu()
        
        # Compare
        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        torch.testing.assert_close(Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-2, rtol=1e-2)
```

### Running Tests

```bash
# With conda kernel (required for Triton)
conda run -n kernel python test_file.py

# For CPU-only testing
conda run -n kernel python test_file.py
```

### Test Template

```python
import torch
import unittest
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "/home/adam.wang/work/vllm")

try:
    from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril as solve_tril_triton
    HAS_TRITON = True
except Exception as e:
    HAS_TRITON = False

def solve_tril_cpu(A: torch.Tensor) -> torch.Tensor:
    """Your CPU implementation here."""
    ...

class TestVerify(unittest.TestCase):
    
    @unittest.skipUnless(HAS_TRITON, "Triton not available")
    def test_cpu_vs_triton_16(self):
        B, T, H = 1, 16, 1
        torch.manual_seed(42)
        A = torch.randn(B, T, H, 16) * 0.1
        A = torch.tril(A, diagonal=-1)
        
        Ai_cpu = solve_tril_cpu(A.clone())
        
        A_gpu = A.clone().cuda()
        Ai_triton = solve_tril_triton(A_gpu)
        Ai_triton_cpu = Ai_triton.cpu()
        
        diff = (Ai_cpu.float() - Ai_triton_cpu.float()).abs().max().item()
        print(f"Max diff: {diff:.6f}")
        torch.testing.assert_close(Ai_cpu.float(), Ai_triton_cpu.float(), atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Causal mask wrong | Use `torch.tril(..., diagonal=-1)` for lower triang (most common) |
| Shape mismatch | Check: `k.shape = [B, T, Hg, K]`, `beta.shape = [B, T, H]` |
| Head mapping | `k_head_idx = h // (H // Hg)` when H != Hg |
| Float32 precision | Allow atol=1e-2, rtol=1e-2 for comparison |
| Block size | Match BT in both implementations |
| Index out of bounds | Use `m_t = o_t < T` mask |
| Triton import fails | Use `sys.path.insert(0, "/home/adam.wang/work/vllm")` and `warnings.filterwarnings('ignore')` |
| Need conda kernel | Use `conda run -n kernel python test_file.py` for GPU testing |

---

