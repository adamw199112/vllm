import numpy as np

def index_trow_numpy(
    batch_offset,
    xrow_offset,
    xcol_offset,
    x_padded_row,
    x_padded_col,
    tH,
    tW,
    XSTEP_ROW,
    XSTEP,
):
    # --- 构造 xrow / xcol（完全对应 Triton 的 broadcast 方式） ---
    xrow = xrow_offset #+ np.arange(XSTEP_ROW)[:, None]   # shape: [XSTEP_ROW, 1]
    xcol = xcol_offset + np.arange(XSTEP) #[None, :]       # shape: [1, XSTEP]

    # --- 按 Triton 原公式逐项计算 ---
    index = (
        batch_offset * x_padded_row * x_padded_col
        + (xrow // tH) * x_padded_col * tH
        + (xcol // tW) * tH * tW
        + (xrow % tH) * tW
        + (xcol % tW)
    )

    return index  # shape: [XSTEP_ROW, XSTEP]



print(index_trow_numpy(0, 1, 0, 4, 128, 4, 32, 1, 128))

print(index_trow_numpy(0, 1, 8, 4, 128, 4, 32, 1, 8))