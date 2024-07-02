from typing import Union

import torch as th
import triton
import triton.language as tl
from torch import Tensor

from fast_gem.functional import triton_utils as tu


@triton.autotune(
    configs=[
        triton.Config({"BLK_L": 4096}),
        triton.Config({"BLK_L": 2048}),
        triton.Config({"BLK_L": 1024}),
        triton.Config({"BLK_L": 512}),
        triton.Config({"BLK_L": 256}),
        triton.Config({"BLK_L": 128}),
        triton.Config({"BLK_L": 64}),
        triton.Config({"BLK_L": 32}),
    ],
    key=["str_x_L"],
)
@triton.jit
def gem_forward_1d_kernel(
    x_ptr,
    y_ptr,
    p,
    eps,
    str_x_B,
    str_x_C,
    str_x_L,
    str_y_B,
    str_y_C,
    L,
    IS_P_TENSOR: tl.constexpr,
    BLK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_l = tl.arange(0, BLK_L)  # l
    x_ptrs = x_ptr + pid_b * str_x_B + pid_c * str_x_C + offs_l * str_x_L

    if IS_P_TENSOR:
        p = tl.load(p)

    y = 0.0
    for idx_n in range(tl.cdiv(L, BLK_L)):
        mask = offs_l < L - idx_n * BLK_L
        x = tl.load(x_ptrs, mask=mask, other=0.0)  # l

        # calculate adaptive average pooling
        x = tl.where((x < eps) & mask, eps, x)
        x = tu.pow(x, p)  # l
        y += tl.sum(x)  # 1

        x_ptrs += BLK_L * str_x_L

    y /= L
    y = tu.pow(y, 1 / p)

    y_ptrs = y_ptr + pid_b * str_y_B + pid_c * str_y_C
    tl.store(y_ptrs, y)


@triton.autotune(
    configs=[
        triton.Config({"BLK_L": 4096}),
        triton.Config({"BLK_L": 2048}),
        triton.Config({"BLK_L": 1024}),
        triton.Config({"BLK_L": 512}),
        triton.Config({"BLK_L": 256}),
        triton.Config({"BLK_L": 128}),
        triton.Config({"BLK_L": 64}),
        triton.Config({"BLK_L": 32}),
    ],
    key=["str_x_L"],
    reset_to_zero=["dp_ptr"],
)
@triton.jit
def gem_backward_1d_kernel(
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
    str_y_B,
    str_y_C,
    L,
    IS_P_TENSOR: tl.constexpr,
    BLK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_l = tl.arange(0, BLK_L)  # l

    x_ptrs = x_ptr + pid_b * str_x_B + pid_c * str_x_C + offs_l * str_x_L
    dx_ptrs = dx_ptr + pid_b * str_x_B + pid_c * str_x_C + offs_l * str_x_L
    y_ptrs = y_ptr + pid_b * str_y_B + pid_c * str_y_C
    dy_ptrs = dy_ptr + pid_b * str_y_B + pid_c * str_y_C

    if IS_P_TENSOR:
        p = tl.load(p)

    # calculate y-level grad
    y = tl.load(y_ptrs)
    dy = tl.load(dy_ptrs)

    if IS_P_TENSOR:
        dp = -tl.log(y) / p * y * dy
    dy = dy / p * y / (tu.pow(y, p) * L)  # m

    # re-calculate x for x-level grad
    for idx_l in range(tl.cdiv(L, BLK_L)):
        mask = offs_l < L - idx_l * BLK_L
        x = tl.load(x_ptrs, mask=mask, other=0.0)  # l

        # calculate adaptive average pooling
        x_ = tl.where((x < eps) & mask, eps, x)
        dx = tl.zeros((BLK_L,), dtype=x.dtype) + dy  # l

        if IS_P_TENSOR:
            dp_tmp = tu.pow(x_, p) * tl.log(x_) * dx
            dp += tl.sum(tl.where(mask, dp_tmp, 0.0))

        dx *= p * tu.pow(x_, p - 1)
        dx = tl.where((x < eps) & mask, 0.0, dx)

        tl.store(dx_ptrs, dx, mask=mask)
        x_ptrs += BLK_L * str_x_L
        dx_ptrs += BLK_L * str_x_L

    if IS_P_TENSOR:
        tl.atomic_add(dp_ptr, dp)


class GeMOps1d(th.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, p: Union[float, Tensor] = 3.0, eps: float = 1e-6, keepdim=True):
        ctx.is_p_tensor = isinstance(p, Tensor)
        assert x.ndim == 3, f"Unknown shape of `x`: {x.shape}"

        B, C, L = x.shape
        y = x.new_empty(list(x.shape[:2]) + ([1] if keepdim else []))  # b c
        str_x_B, str_x_C, str_x_L = x.stride()
        str_y_B, str_y_C = y.stride(0), y.stride(1)

        grid = lambda meta: (B, C)
        gem_forward_1d_kernel[grid](x, y, p, eps, str_x_B, str_x_C, str_x_L, str_y_B, str_y_C, L, IS_P_TENSOR=ctx.is_p_tensor)

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

        B, C, L = x.shape
        str_x_B, str_x_C, str_x_L = x.stride()
        str_y_B, str_y_C = y.stride()

        dx = th.empty_like(x)
        dp = None
        if ctx.is_p_tensor:
            dp = th.zeros_like(p)

        grid = lambda meta: (B, C)
        gem_backward_1d_kernel[grid](
            x, y, p, dx, dy, dp, eps, str_x_B, str_x_C, str_x_L, str_y_B, str_y_C, L, IS_P_TENSOR=ctx.is_p_tensor
        )
        return dx, dp, None, None, None


def gem_ops1d(x: Tensor, p: Union[float, Tensor] = 3.0, eps: float = 1e-6, keepdim=True):
    return GeMOps1d.apply(x, p, eps, keepdim)
