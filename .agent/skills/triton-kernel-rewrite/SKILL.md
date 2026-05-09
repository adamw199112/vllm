---
name: triton-kernel-rewrite
description: Rewrite Triton kernels from parallel execution to single-core execution with internal loops. Use when user wants to modify Triton kernels, simplify kernel launches, remove autotune heuristics, or convert parallel kernels to use grid=(1,1) with loops.
---

# Triton Kernel Rewrite

Convert a Triton kernel from parallel execution (multiple programs) to single-core execution with internal loops. This is useful for debugging, simplification, or when parallelism isn't needed.

**IMPORTANT**: This skill does not require any vllm imports. Use only `import torch` and `import triton`. If you need utility functions, copy them into the file.

## When to use this skill

- User asks to rewrite a Triton kernel
- User wants to remove `triton.heuristics` or `triton.autotune` decorators
- User wants to simplify kernel execution to single core
- User wants to convert parallel programs into loops inside the kernel
- User asks to remove environment variable configurations
- User wants to ensure all parameters are passed (no None defaults)
- User specifies to remove vllm dependencies
- User asks to package kernel and test case to a directory

## The Rewrite Process

### Step 1: Read and analyze the original kernel

Before making any changes, understand:
- What parameters are passed to the kernel
- How the kernel uses `program_id` to split work
- What heuristics/autotune decorators exist
- What environment configurations are used

### Step 2: Remove vllm imports and dependencies

Replace vllm-specific imports:
```python
# REMOVE THESE:
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices
from vllm.model_executor.layers.fla.ops.utils import input_guard, is_amd, is_tma_supported

# REPLACE WITH:
import triton
import triton.language as tl
```

If `prepare_chunk_indices` is needed, copy it into the file:
```python
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Local implementation of prepare_chunk_indices."""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
```

For `is_tma_supported`, use a simple value:
```python
is_tma_supported = False  # Or True based on your hardware
```

**Remove any `@input_guard` decorator** - simply delete it, it's not needed.

### Step 3: Remove decorations

Remove these if present:
- `@triton.heuristics({...})`
- `@triton.autotune(...)`
- Any `FLA_TRIL_PRECISION` or similar env var handling

### Step 4: Change data type from float32 to float16

In kernel:
- Change `.to(tl.float32)` to `.to(tl.float16)`
- Change any explicit float32 references to float16

In test case:
- Change input data creation to use `dtype=torch.float16`
- Change any assertions to use appropriate tolerances for float16

### Step 5: Ensure parameter completeness

Every parameter passed to the kernel must have a value:
- If `cu_seqlens` can be None, create a dummy tensor when None
- If `chunk_indices` can be None, create appropriate default
- Pass all required values explicitly, don't rely on heuristics

### Step 6: Convert to single-core grid

Change kernel launch from:
```python
kernel[NT, B * H](...)  # Parallel
```

To:
```python
kernel[1, 1](...)  # Single core
```

### Step 7: Add loop parameters

Since we're using single core, pass the loop bounds as constexpr:
```python
B: tl.constexpr,
NT: tl.constexpr,
```

### Step 8: Rewrite kernel body

Replace `program_id` based parallelism with loops:

**Before (parallel):**
```python
i_t, i_bh = tl.program_id(0), tl.program_id(1)
i_b, i_h = i_bh // H, i_bh % H
```

**After (single-core with loops):**
```python
for i_bh in range(B * H):
    i_b = i_bh // H
    i_h = i_bh % H
    for i_t in range(NT):
        # ... original logic
```

### Step 9: Handle Triton limitations

Triton JIT has restrictions:
- **No `break` in loops** - Compute loop bound before entering loop
- **No dynamic ranges in some contexts** - Use `tl.constexpr` for known values
- **Pointer shapes** - Some operations require specific shapes

### Step 10: Package to separate directory

Create a directory (e.g., `./op_release/`) and save:
- The rewritten kernel file (e.g., `solve_tril.py`)
- The test case file (e.g., `test_solve_tril.py`)

Both files should be self-contained with no external dependencies.

### Step 11: Create standalone test case

The test case should have these imports:
```python
import torch
import unittest
import pytest
import sys
import warnings
```

All other imports (vllm, etc.) must be removed. Any utility functions needed should be defined in the test file itself.

## Common patterns

### Pattern: Convert IS_VARLEN handling

**Before:**
```python
if IS_VARLEN:
    i_n = tl.load(chunk_indices + i_t * 2)...
```

**After:** Keep the same logic inside the loop, pass IS_VARLEN as constexpr

### Pattern: Ensure cu_seqlens is never None

```python
cu_seqlens_ptr = cu_seqlens if cu_seqlens is not None else torch.zeros(2, dtype=torch.int32, device=A.device)
```

### Pattern: Handle loop termination without break

**Before (invalid in Triton):**
```python
for i in range(2, BT):
    if row_idx >= T_val:
        break  # NOT ALLOWED
```

**After (compute bound):**
```python
max_iter = min(BT, T_val - chunk_idx * BT)
for i in range(2, max_iter):
    # ... valid
```

### Pattern: Dummy chunk_indices for non-varlen

```python
if not is_varlen:
    NT = triton.cdiv(T, BT)
    dummy_chunk_indices = torch.zeros((NT, 2), dtype=torch.int32, device=A.device)
    for i in range(NT):
        dummy_chunk_indices[i, 0] = 0
        dummy_chunk_indices[i, 1] = i
    chunk_indices_ptr = dummy_chunk_indices
```

### Pattern: Replace vllm-specific triton import

```python
# Original (REMOVE):
from vllm.triton_utils import tl, triton

# New (USE THIS):
import triton
import triton.language as tl
```

### Pattern: Inline utility functions

If the original code uses `prepare_chunk_indices`:
```python
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute chunk indices for variable length sequences."""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
```

### Pattern: Float16 data type

In kernel:
```python
# Change from:
b_A = tl.load(base_A).to(tl.float32)
# To:
b_A = tl.load(base_A).to(tl.float16)
```

In test case:
```python
# Change from:
A = torch.randn(B, T, H, BT, dtype=torch.float32) * 0.1
# To:
A = torch.randn(B, T, H, BT, dtype=torch.float16) * 0.1
```

## Example transformation

Original with vllm imports:
```python
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices
from vllm.model_executor.layers.fla.ops.utils import input_guard, is_tma_supported

FLA_TRIL_PRECISION = os.environ.get("FLA_TRIL_PRECISION", "ieee")

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(configs=[...], key=["BT"])
@triton.jit(do_not_specialize=["T"])
def solve_tril_kernel(A, Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    ...

# Launch:
solve_tril_kernel[NT, B * H](...)
```

Rewritten (self-contained in ./op_release/):

**solve_tril.py:**
```python
import torch
import triton
import triton.language as tl

def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute chunk indices for variable length sequences."""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in num_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

@triton.jit
def solve_tril_kernel(A, Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION, B, NT):
    for i_bh in range(B * H):
        for i_t in range(NT):
            # Use float16:
            b_A = tl.load(base_A).to(tl.float16)
            ...

# Launch:
solve_tril_kernel[1, 1](...)
```

**test_solve_tril.py:**
```python
import torch
import unittest
import sys
import warnings

warnings.filterwarnings("ignore")

# Define any needed utility functions here
def prepare_chunk_indices(cu_seqlens, chunk_size):
    ...

def solve_tril_cpu(A):
    ...  # CPU implementation using float16

class TestSolveTril(unittest.TestCase):
    def test_float16(self):
        A = torch.randn(1, 16, 1, 16, dtype=torch.float16) * 0.1
        ...
```

## Notes

- This transformation trades parallelism for simplicity
- Performance will likely decrease (single core vs multiple)
- Suitable for debugging, testing, or small inputs
- Always verify correctness with existing test cases
- Keep imports minimal: only `torch` and `triton`
- Copy any needed utility functions into the file
- Package both kernel and test to the same directory
- Use float16 for all data types