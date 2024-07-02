from typing import Union

import torch as th
import triton
import triton.language as tl
from torch import Tensor

from fast_gem.functional import triton_utils as tu


@triton.autotune(
    configs=[
        triton.Config({"BLK_L": 256, "BLK_M": 32, "BLK_N": 32}),
        triton.Config({"BLK_L": 128, "BLK_M": 64, "BLK_N": 32}),
        triton.Config({"BLK_L": 128, "BLK_M": 32, "BLK_N": 64}),
        triton.Config({"BLK_L": 64, "BLK_M": 128, "BLK_N": 32}),
        triton.Config({"BLK_L": 64, "BLK_M": 64, "BLK_N": 64}),
        triton.Config({"BLK_L": 64, "BLK_M": 32, "BLK_N": 128}),
        triton.Config({"BLK_L": 32, "BLK_M": 256, "BLK_N": 32}),
        triton.Config({"BLK_L": 32, "BLK_M": 128, "BLK_N": 64}),
        triton.Config({"BLK_L": 32, "BLK_M": 64, "BLK_N": 128}),
        triton.Config({"BLK_L": 32, "BLK_M": 32, "BLK_N": 256}),
    ],
    key=["str_x_L", "str_x_M", "str_x_N"],
)
@triton.jit
def gem_forward_3d_kernel(
    x_ptr,
    y_ptr,
    p,
    eps,
    str_x_B,
    str_x_C,
    str_x_L,
    str_x_M,
    str_x_N,
    str_y_B,
    str_y_C,
    L,
    M,
    N,
    IS_P_TENSOR: tl.constexpr,
    BLK_L: tl.constexpr,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_l = tl.arange(0, BLK_L)  # l
    offs_m = tl.arange(0, BLK_M)  # m
    offs_n = tl.arange(0, BLK_N)  # m
    x_ptrs = (
        x_ptr
        + pid_b * str_x_B
        + pid_c * str_x_C
        + offs_l[:, None, None] * str_x_L
        + offs_m[None, :, None] * str_x_M
        + offs_n[None, None, :] * str_x_N
    )

    if IS_P_TENSOR:
        p = tl.load(p)

    y = 0.0
    for idx_l in range(tl.cdiv(L, BLK_L)):
        mask_l = offs_l[:, None, None] < L - idx_l * BLK_L
        for idx_m in range(tl.cdiv(M, BLK_M)):
            mask_m = offs_m[None, :, None] < M - idx_m * BLK_M
            for idx_n in range(tl.cdiv(N, BLK_N)):
                mask = mask_l & mask_m & (offs_n[None, None, :] < N - idx_n * BLK_N)

                x = tl.load(x_ptrs, mask=mask, other=0.0)  # l m

                # calculate adaptive average pooling
                x = tl.where((x < eps) & mask, eps, x)
                x = tu.pow(x, p)  # l
                y += tl.sum(x)  # 1

                x_ptrs += BLK_N * str_x_N
            x_ptrs += BLK_M * str_x_M
        x_ptrs += BLK_L * str_x_L

    y /= L * M * N
    y = tu.pow(y, 1 / p)

    y_ptrs = y_ptr + pid_b * str_y_B + pid_c * str_y_C
    tl.store(y_ptrs, y)


@triton.autotune(
    configs=[
        triton.Config({"BLK_L": 256, "BLK_M": 32, "BLK_N": 32}),
        triton.Config({"BLK_L": 128, "BLK_M": 64, "BLK_N": 32}),
        triton.Config({"BLK_L": 128, "BLK_M": 32, "BLK_N": 64}),
        triton.Config({"BLK_L": 64, "BLK_M": 128, "BLK_N": 32}),
        triton.Config({"BLK_L": 64, "BLK_M": 64, "BLK_N": 64}),
        triton.Config({"BLK_L": 64, "BLK_M": 32, "BLK_N": 128}),
        triton.Config({"BLK_L": 32, "BLK_M": 256, "BLK_N": 32}),
        triton.Config({"BLK_L": 32, "BLK_M": 128, "BLK_N": 64}),
        triton.Config({"BLK_L": 32, "BLK_M": 64, "BLK_N": 128}),
        triton.Config({"BLK_L": 32, "BLK_M": 32, "BLK_N": 256}),
    ],
    key=["str_x_L", "str_x_M", "str_x_N"],
    reset_to_zero=["dp_ptr"],
)
@triton.jit
def gem_backward_3d_kernel(
    x_ptr,
    y_ptr,
    p,
    dx_ptr,
    dy_ptr,
    dp_ptr,
    eps,
    str_x_B,
    str_x_C,
    str_x_L,
    str_x_M,
    str_x_N,
    str_y_B,
    str_y_C,
    L,
    M,
    N,
    IS_P_TENSOR: tl.constexpr,
    BLK_L: tl.constexpr,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_l = tl.arange(0, BLK_L)  # l
    offs_m = tl.arange(0, BLK_M)  # m
    offs_n = tl.arange(0, BLK_N)  # m

    offs_x = (
        pid_b * str_x_B
        + pid_c * str_x_C
        + offs_l[:, None, None] * str_x_L
        + offs_m[None, :, None] * str_x_M
        + offs_n[None, None, :] * str_x_N
    )
    x_ptrs = x_ptr + offs_x
    dx_ptrs = dx_ptr + offs_x
    y_ptrs = y_ptr + pid_b * str_y_B + pid_c * str_y_C
    dy_ptrs = dy_ptr + pid_b * str_y_B + pid_c * str_y_C

    if IS_P_TENSOR:
        p = tl.load(p)

    # calculate y-level grad
    y = tl.load(y_ptrs)
    dy = tl.load(dy_ptrs)

    if IS_P_TENSOR:
        dp = -tl.log(y) / p * y * dy
    dy = dy / p * y / (tu.pow(y, p) * L * M * N)

    # re-calculate x for x-level grad
    for idx_l in range(tl.cdiv(L, BLK_L)):
        mask_l = offs_l[:, None, None] < L - idx_l * BLK_L
        for idx_m in range(tl.cdiv(M, BLK_M)):
            mask_m = offs_m[None, :, None] < M - idx_m * BLK_M
            for idx_n in range(tl.cdiv(N, BLK_N)):
                mask = mask_l & mask_m & (offs_n[None, None, :] < N - idx_n * BLK_N)
                x = tl.load(x_ptrs, mask=mask, other=0.0)  # l

                # calculate adaptive average pooling
                x_ = tl.where(mask & (x < eps), eps, x)
                x_p1 = tu.pow(x_, p - 1)
                dx = tl.zeros((BLK_L, BLK_M, BLK_N), dtype=x.dtype) + dy  # l m

                if IS_P_TENSOR:
                    dp_tmp = tl.where(mask, x_p1 * x_ * tl.log(x_) * dx, 0.0)
                    dp += tl.sum(dp_tmp)

                dx *= p * x_p1
                dx = tl.where((x < eps) & mask, 0.0, dx)

                tl.store(dx_ptrs, dx, mask=mask)
                x_ptrs += BLK_N * str_x_N
                dx_ptr += BLK_N * str_x_N
            x_ptrs += BLK_M * str_x_M
            dx_ptrs += BLK_M * str_x_M
        x_ptrs += BLK_L * str_x_L
        dx_ptrs += BLK_L * str_x_L

    if IS_P_TENSOR:
        tl.atomic_add(dp_ptr, dp)


class GeMOps3d(th.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, p: Union[float, Tensor] = 3.0, eps: float = 1e-6, keepdim=True):
        ctx.is_p_tensor = isinstance(p, Tensor)
        assert x.ndim == 5, f"Unknown shape of `x`: {x.shape}"

        B, C, L, M, N = x.shape
        y = x.new_empty(list(x.shape[:2]) + ([1, 1, 1] if keepdim else []))  # b c
        # str_x_B, str_x_C, str_x_L, str_x_M, str_x_N = x.stride()
        # str_y_B, str_y_C = y.stride(0), y.stride(1)
        strides = (*x.stride(), *y.stride()[:2])

        grid = lambda meta: (B, C)
        gem_forward_3d_kernel[grid](x, y, p, eps, *strides, L, M, N, IS_P_TENSOR=ctx.is_p_tensor)

        if ctx.is_p_tensor:
            ctx.save_for_backward(x, p, y)
            ctx.params = (eps,)
        else:
            ctx.save_for_backward(x, y)
            ctx.params = p, eps
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        if ctx.is_p_tensor:
            x, p, y = ctx.saved_tensors
            (eps,) = ctx.params
        else:
            x, y = ctx.saved_tensors
            p, eps = ctx.params

        B, C, L, M, N = x.shape
        # str_x_B, str_x_C, str_x_L, str_x_M, str_x_N = x.stride()
        # str_y_B, str_y_C = y.stride()
        strides = (*x.stride(), *y.stride()[:2])

        dx = th.empty_like(x)
        dp = None
        if ctx.is_p_tensor:
            dp = th.zeros_like(p)

        grid = lambda meta: (B, C)
        gem_backward_3d_kernel[grid](x, y, p, dx, dy, dp, eps, *strides, L, M, N, IS_P_TENSOR=ctx.is_p_tensor)
        return dx, dp, None, None, None


def gem_ops3d(x: Tensor, p: Union[float, Tensor] = 3.0, eps: float = 1e-6, keepdim=True):
    return GeMOps3d.apply(x, p, eps, keepdim)
