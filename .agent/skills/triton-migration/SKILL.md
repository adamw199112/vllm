---
name: triton-migration
description: "Migrate Triton kernels between `tl.make_block_ptr` and `offsets + tl.arange` indexing patterns. Use this skill when the user wants to convert Triton code between block pointer API and manual offset calculation, including both forward migration (block_ptr → offsets+arange) and reverse migration (offsets+arange → block_ptr). Trigger whenever the user mentions 'block_ptr', 'make_block_ptr', 'tl.arange' migration, conversion, or rewriting Triton indexing."
---

# Triton Indexing Migration Skill

Migrate Triton kernels between two indexing patterns:
1. **Forward**: `tl.make_block_ptr` → `offsets + tl.arange`
2. **Reverse**: `offsets + tl.arange` → `tl.make_block_ptr`

---

## Quick Reference

| Direction | Command |
|-----------|---------|
| block_ptr → offsets | See [Forward Migration](#forward-migration-block_ptr--to-offsets--arange) |
| offsets → block_ptr | See [Reverse Migration](#reverse-migration-offsets--arange-to-block_ptr) |

---

## Forward Migration: block_ptr → offsets + arange

### When to Use

The user wants to convert from `tl.make_block_ptr` API to manual offset calculation with `tl.arange`.

### Critical Rule: Keep block_ptr for Direct tl.dot Inputs

**IMPORTANT**: Check whether the tensor is used directly as input to `tl.dot`:
- **Keep block_ptr**: If the tensor is used directly as argument to `tl.dot` (e.g., `tl.dot(b_tensor, ...)` or `tl.dot(..., b_tensor)`) **without any transformation on that tensor**
- **Convert to offset+arange**: If the tensor is NOT used directly in tl.dot (e.g., used in subtraction, addition, store, or other operations)

**Correct analysis - keep block_ptr for dot inputs**:
```python
# k is used directly in dot -> KEEP block_ptr
p_k = tl.make_block_ptr(k, ...)
b_k = tl.load(p_k)
b_h += tl.dot(b_k, b_v)  # b_k directly in dot

# w is used directly in dot -> KEEP block_ptr  
p_w = tl.make_block_ptr(w, ...)
b_w = tl.load(p_w)
b_v += tl.dot(b_w, tl.trans(b_h))  # b_w directly in dot

# v is NOT used in dot (subtraction) -> CONVERT to offset+arange
p_v = tl.make_block_ptr(v, ...)
b_v = tl.load(p_v)
b_v = b_v - b_v_delta  # v NOT directly in dot

# v_new is NOT used in dot (store only) -> CONVERT to offset+arange
p_v = tl.make_block_ptr(v_new, ...)
tl.store(p_v, b_v)  # v_new NOT directly in dot
```

**Common mistake to avoid**:
```python
# WRONG: Converting k and w because they ARE used in dot
p_k = tl.make_block_ptr(k, ...)
b_k = tl.load(p_k)
b_o = tl.dot(b_k, b_v)  # k IS used directly in dot - should KEEP block_ptr!

# WRONG: Keeping v because "it's used somewhere"
p_v = tl.make_block_ptr(v, ...)
b_v = tl.load(p_v)
b_v = b_v - other  # v NOT in dot - should CONVERT to offset+arange!
```

### Pattern Recognition

Identify `tl.make_block_ptr` calls:
```python
p = tl.make_block_ptr(
    base + offset,      # base address with base offset
    (T,),             # shape
    (S,),              # stride
    (i_t * BT,),       # block offset
    (BT,),             # block shape
    (0,)               # order
)
```
**Fix for mask**: Always check both dimensions for 2D store:
```python
# Must match the load mask
u_mask = (t_offsets[:, None] < T) & (v_idx_offsets[None, :] < V)
tl.store(o + u_offsets, b_o.to(tl.float32), mask=u_mask)
```

---

## Case Study: chunk_fwd_kernel_o Migration

Real-world example migrating `chunk_fwd_kernel_o` from block_ptr to offset+arange.

### Original Code
```python
# p_h - NOT used directly in dot (used in tl.trans before dot)
p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
b_h = tl.load(p_h, boundary_check=(0, 1))
b_o += tl.dot(b_q, tl.trans(b_h))  # b_h is result of trans, not direct dot input

# p_v - NOT used in dot (result of subtraction)
p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
b_v = tl.load(p_v, boundary_check=(0, 1))
b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale  # b_v is operand to subtraction, not in dot

# p_o - output store
p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

# p_q, p_k - USED directly in dot
p_q = tl.make_block_ptr(q, ...)
p_k = tl.make_block_ptr(k, ...)
b_q = tl.load(p_q)
b_k = tl.load(p_k)
b_o = tl.dot(b_q, b_k)  # both used directly in dot!
```

### Migration Results

| Tensor | Original | Converted To | Reason |
|--------|----------|--------------|--------|
| `p_h` | block_ptr | offset+arange | h NOT used in dot (used in tl.trans before dot) |
| `p_v` | block_ptr | offset+arange | v NOT used directly in dot (result of subtraction) |
| `p_o` | block_ptr | offset+arange | Store output |
| `p_q`, `p_k` | **keep block_ptr** | block_ptr | Used directly in tl.dot |

### Converted Code
```python
# p_h -> offset+arange (NOT used directly in dot)
t_offsets_h = i_v * BV + tl.arange(0, BV)
k_offsets = i_k * BK + tl.arange(0, BK)
h_offsets = t_offsets_h[:, None] * K + k_offsets[None, :]
h_mask = (t_offsets_h[:, None] < V) & (k_offsets[None, :] < K)
b_h = tl.load(h + h_offsets, mask=h_mask, other=0.0)

# p_v -> offset+arange (NOT used directly in dot - result of subtraction)
t_offsets_v = i_t * BT + tl.arange(0, BT)
v_idx_offsets = i_v * BV + tl.arange(0, BV)
v_offsets = t_offsets_v[:, None] * H * V + v_idx_offsets[None, :]
v_mask = (t_offsets_v[:, None] < T) & (v_idx_offsets[None, :] < V)
b_v = tl.load(v + v_offsets, mask=v_mask, other=0.0)

# p_o -> offset+arange (store output)
t_offsets_o = i_t * BT + tl.arange(0, BT)
v_idx_offsets = i_v * BV + tl.arange(0, BV)
o_offsets_full = t_offsets_o[:, None] * H * V + v_idx_offsets[None, :]
o_mask = (t_offsets_o[:, None] < T) & (v_idx_offsets[None, :] < V)

b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
tl.store(o + o_offsets_full, b_o.to(tl.float32), mask=o_mask)

# p_q, p_k -> KEEP block_ptr (USED directly in dot)
p_q = tl.make_block_ptr(q, ...)
b_q = tl.load(p_q, boundary_check=(0, 1))
p_k = tl.make_block_ptr(k, ...)
b_k = tl.load(p_k, boundary_check=(0, 1))
b_o = tl.dot(b_q, b_k)  # both used directly in dot!
```

### Key Lessons

1. **Check if used directly in dot**: If tensor is argument to tl.dot, keep block_ptr; if used elsewhere (add, subtract, store, etc.), convert to offset+arange
2. **2D load needs 2D mask**: `(t_mask) & (v_idx_mask)`
3. **2D store needs 2D mask**: Same as load mask, not just first dimension
4. **Use tl.float32 for store**: Don't rely on pointer's element_ty, use explicit `.to(tl.float32)`

### Migration Steps

1. **Extract base address** from first argument of `tl.make_block_ptr`
2. **Extract strides** from the stride arguments
3. **Calculate offsets** using `i_t * BT + tl.arange(0, BT)` pattern
4. **Compute base addresses** by adding offsets to base
5. **Replace load/store** with direct indexing (no `boundary_check`)

### Example Transformation

**Before (block_ptr):**
```python
p_v = tl.make_block_ptr(
    v + (bos * H + i_h) * V,
    (T, V),
    (H * V, 1),
    (i_t * BT, i_v * BV),
    (BT, BV),
    (1, 0),
)
p_u = tl.make_block_ptr(
    u + (bos * H + i_h) * V,
    (T, V),
    (H * V, 1),
    (i_t * BT, i_v * BV),
    (BT, BV),
    (1, 0),
)
b_v = tl.load(p_v, boundary_check=(0, 1))
b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
b_u = tl.dot(b_A, b_vb, allow_tf32=False)
tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
```

**After (offsets + arange):**
```python
# Load v with offset+arange
t_offsets = i_t * BT + tl.arange(0, BT)
v_base = (bos * H + i_h) * V
v_offsets = v_base + t_offsets[:, None] * H * V + (i_v * BV + tl.arange(0, BV))[None, :]
v_mask = (t_offsets[:, None] < T) & ((i_v * BV + tl.arange(0, BV))[None, :] < V)
b_v = tl.load(v + v_offsets, mask=v_mask, other=0.0)
b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)

# Compute dot
b_u = tl.dot(b_A, b_vb, allow_tf32=False)

# Store u with offset+arange
u_base = (bos * H + i_h) * V
u_offsets = u_base + t_offsets[:, None] * H * V + (i_v * BV + tl.arange(0, BV))[None, :]
u_mask = (t_offsets[:, None] < T) & ((i_v * BV + tl.arange(0, BV))[None, :] < V)
tl.store(u + u_offsets, b_u.to(tl.float32), mask=u_mask)
```

### Handling Different Data Layouts

**head_first (row-major in first dim):**
```python
base = base_addr + bos * H + i_h * T + offsets
```

**non-head-first (contiguous in last dim):**
```python
base = base_addr + bos * H + i_h + offsets * H
```

For 2D block pointers:
```python
# block_ptr form:
tl.make_block_ptr(base, (T, S), (S, 1), (i_t*BT, i_s*BS), (BT, BS), (1, 0))

# offsets form:
offsets_t = i_t * BT + tl.arange(0, BT)
offsets_s = i_s * BS + tl.arange(0, BS)
base = base_addr + offsets_t * S + offsets_s  # for S stride
# or:
base = base_addr + offsets_t + offsets_s * S  # depending on layout
```

---

## 2D Masking: Critical Lesson Learned

### The Problem

When converting 2D block_ptr with `boundary_check=(0, 1)` to offset+arange, the mask must check **BOTH dimensions**:

```python
# WRONG - only checks t dimension
v_mask = t_offsets[:, None] < T  # This causes test failures!

# CORRECT - checks both t and v_idx dimensions
v_mask = (t_offsets[:, None] < T) & ((i_v * BV + tl.arange(0, BV))[None, :] < V)
```

### Why This Matters

The original `boundary_check=(0, 1)` ensures:
- Rows (t dimension): indices >= shape[0] are masked out
- Columns (v_idx dimension): indices >= shape[1] are masked out

Failing to check the second dimension causes garbage data to contaminate results, leading to test failures with large numerical differences.

### Complete Mask Pattern for 2D Load/Store

```python
# For 2D load (reading data)
t_offsets = i_t * BT + tl.arange(0, BT)
t_mask = t_offsets[:, None] < T  # t dimension check

# Only add second dimension check if BV might exceed V
if BV < V:
    v_idx_mask = (i_v * BV + tl.arange(0, BV))[None, :] < V
    v_mask = t_mask & v_idx_mask
else:
    v_mask = t_mask

b_v = tl.load(v + v_offsets, mask=v_mask, other=0.0)

# For store (writing output), must check BOTH dimensions!
# The block size (BT, BV) may exceed the actual tensor size (T, V)
u_mask = (t_offsets[:, None] < T) & (v_idx_offsets[None, :] < V)
tl.store(u + u_offsets, b_u.to(tl.float32), mask=u_mask)
```

### Critical Fix for 2D Store

**IMPORTANT**: A common mistake is only checking the first dimension for store:
```python
# WRONG - causes test failures!
u_mask = t_offsets[:, None] < T
tl.store(o + u_offsets, b_o, mask=u_mask)

# CORRECT - checks both dimensions
u_mask = (t_offsets[:, None] < T) & (v_idx_offsets[None, :] < V)
tl.store(o + u_offsets, b_o.to(tl.float32), mask=u_mask)
```

---

## Offset Formula Construction

### Multi-dimensional Tensor Flat Index

For tensor `t[b, i, j, k]` with stride `(S0, S1, S2, S3)`:
```python
flat_index = b*S0 + i*S1 + j*S2 + k*S3
```

### 2D Tensor Offset Formula

For tensor `v[b, t, h, v_idx]` with shape `(B, T, H, V)`:
```python
# block_ptr: base + off0 * stride0 + off1 * stride1
# where stride0 = H*V, stride1 = 1 (for order=(1,0))

# offsets form:
t_offsets = i_t * BT + tl.arange(0, BT)  # [BT,]
v_offsets = (i_v * BV + tl.arange(0, BV))  # [BV,]

# Flat offset = t_part + v_part
# t_part = (bos + t_offsets) * H * V  # scale t by H*V
# v_part = v_offsets  # v_idx is innermost
full_offsets = ((bos + t_offsets)[:, None] * H * V + v_offsets[None, :])
```

### Verification Method

Always verify offset formula against block_ptr:
```python
# block_ptr flat index:
# base + (i_t*BT + off0) * H*V + (i_v*BV + off1)

# offset+arange flat index:
# base + (bos + i_t*BT + off0) * H*V + (i_v*BV + off1)
# where off0 from tl.arange(0,BT), off1 from tl.arange(0,BV)
```

---

## Incremental Testing Strategy

### Why Incremental Testing?

Converting all block_ptr at once can make debugging difficult. Test incrementally:

1. **Start with one tensor** (e.g., v, u) - verify it matches
2. **Then add more** (e.g., k, w) - verify after each addition
3. **Finally, clean up** any redundant variables

### Testing Command
```bash
# Clear Triton cache before testing
rm -rf ~/.triton/cache

# Run tests
python test_file.py -v
```

### Test Pattern
```python
# Isolated test for single tensor conversion
@triton.jit
def test_single_tensor_block_ptr(data, out, H, V, T, BT, BV):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = 0, i_bh % H
    bos = i_b * T
    
    p_data = tl.make_block_ptr(
        data + (bos * H + i_h) * V,
        (T, V), (H * V, 1),
        (i_t * BT, 0), (BT, BV), (1, 0)
    )
    b_data = tl.load(p_data, boundary_check=(0, 1))
    tl.store(p_data, b_data, boundary_check=(0, 1))

@triton.jit
def test_single_tensor_offset(data, out, H, V, T, BT, BV):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = 0, i_bh % H
    bos = i_b * T
    
    t_offsets = i_t * BT + tl.arange(0, BT)
    offsets = (bos * H + i_h) * V + t_offsets[:, None] * H * V + tl.arange(0, BV)[None, :]
    mask = t_offsets[:, None] < T
    b_data = tl.load(data + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, b_data, mask=mask)

# Verify they produce identical results
```

---

## Reverse Migration: offsets + arange → block_ptr

### When to Use

The user wants to convert from `offsets + tl.arange` pattern back to `tl.make_block_ptr` API.

### Pattern Recognition

Identify manual offset patterns:
```python
offsets = i_t * BT + tl.arange(0, BT)
base = base_addr + offsets
data = tl.load(base)
```
Note: This is less common since block_ptr provides automatic boundary checking.

### Migration Steps

1. **Identify base expression** that combines base address + offsets
2. **Extract stride** from how offsets are applied (multiplicative vs additive)
3. **Reconstruct block_ptr** arguments:
   - **base**: The base address without offsets
   - **shape**: The shape of the loaded data
   - **stride**: The stride between elements
   - **block_offset**: The i_t * BT part
   - **block_shape**: The BT part
   - **order**: Layout order (0 for head-first, 1 for non-head-first)

### Example Transformation

**Before (offsets + arange):**
```python
offsets = i_t * BT + tl.arange(0, BT)
base_s = s + bos * H + i_h * T + offsets
b_s = tl.load(base_s)
```

**After (block_ptr):**
```python
p_s = tl.make_block_ptr(
    s + bos * H + i_h * T,
    (T,),
    (1,),
    (i_t * BT,),
    (BT,),
    (0,)
)
b_s = tl.load(p_s, boundary_check=(0,))
```

### Reverse 2D Migration

**Before:**
```python
offsets_t = i_t * BT + tl.arange(0, BT)
offsets_s = i_s * BS + tl.arange(0, BS)
base = base_addr + offsets_t * S + offsets_s
```

**After:**
```python
p = tl.make_block_ptr(
    base_addr,
    (T, S),
    (S, 1),
    (i_t * BT, i_s * BS),
    (BT, BS),
    (1, 0),
)
```

---

## Finding Target Kernels

### Search for block_ptr Usage
```bash
grep -rn "tl.make_block_ptr" <project_root>
```

### Search for offsets + arange Pattern
```bash
grep -rn "tl.arange" <project_root> | grep -v ".pyc"
```

---

## Testing

Create isolated test scripts to verify correctness:

```python
import sys
sys.path.insert(0, "<project_root>")
import triton
import triton.language as tl

# Copy kernel code to test file
# Run with:
pytest test_file.py -v
```

Or for quick testing:
```bash
python test_file.py
```

---

## Common Patterns Reference

| Pattern | block_ptr form | offsets form |
|---------|----------------|-------------|
| 1D head-first | `tl.make_block_ptr(base, (T,), (S,), (i_t*BT,), (BT,), (0,))` | `base = base_addr + i_t*BT + tl.arange(0, BT)` |
| 1D non-head | `tl.make_block_ptr(base, (T,), (H,), (i_t*BT,), (BT,), (0,))` | `base = base_addr + i_t*BT + tl.arange(0, BT)*H` |
| 2D | `tl.make_block_ptr(base, (T,S), (S,1), (i_t*BT,i_s*BS), (BT,BS), (1,0))` | `base = base_addr + (i_t*BT+tl.arange(0,BT))*S + (i_s*BS+tl.arange(0,BS))` |

---

## Troubleshooting

### Issue: Large numerical differences after conversion

**Cause**: 2D mask only checks one dimension

**Fix**: Ensure both dimensions are checked:
```python
mask = (t_offsets[:, None] < T) & (v_offsets < V)
```

### Issue: Test passes for large T but fails for small T

**Cause**: Autotune picking different configurations for different sizes

**Fix**: Clear Triton cache before testing:
```bash
rm -rf ~/.triton/cache
```

### Issue: Off-by-one errors in output

**Cause**: Incorrect offset formula construction

**Fix**: Verify against block_ptr formula. For 4D tensor `v[b,t,h,v_idx]`:
- block_ptr: `base + (i_t*BT+off0)*H*V + (i_v*BV+off1)`
- offset: `(bos+t_offsets)*H*V + v_offsets`

### Issue: NaN or inf values in output

**Cause**: Mask not properly applied, garbage data entering computation

**Fix**: Double-check mask dimensions match tensor shape

### Issue: Store using block_ptr works but offset+arange fails

**Cause**: Incorrect dtype conversion in store, or mask only checking one dimension

**Fix for dtype**: Use `tl.float32` for store instead of relying on pointer's dtype:
```python
# WRONG - may fail
tl.store(o + u_offsets, b_o.to(b_v.dtype), mask=u_mask)

# CORRECT - use tl.float32 explicitly
tl.store(o + u_offsets, b_o.to(tl.float32), mask=u_mask)
```

**Fix for mask**: Always check both dimensions for 2D store:
```python
# Must match the load mask
u_mask = (t_offsets[:, None] < T) & (v_idx_offsets[None, :] < V)
tl.store(o + u_offsets, b_o.to(tl.float32), mask=u_mask)
```