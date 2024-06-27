from typing import Union

import torch as th
import triton
import triton.language as tl
from torch import Tensor

from fast_gem.functional import triton_utils as tu


class GeMOps(th.autograd.Function):
    @staticmethod
    @triton.jit
    def forward_kernel(x_ptr, y_ptr, p, eps, M, N, IS_P_TENSOR: tl.constexpr, BLK_M: tl.constexpr, BLK_N: tl.constexpr):
        pid = tl.program_id(0)
        offs_m = pid * BLK_M + tl.arange(0, BLK_M)  # m
        offs_n = tl.arange(0, BLK_N)  # n
        x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]  # m n

        if IS_P_TENSOR:
            p = tl.load(p)

        y = tl.zeros((BLK_M,), dtype=tl.float32)
        for n in range(tl.cdiv(N, BLK_N)):
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N - n * BLK_N)
            x = tl.load(x_ptrs, mask=mask, other=0.0)  # m n

            # calculate adaptive average pooling
            x = tl.where((x < eps) & mask, eps, x)
            x = tu.pow(x, p)  # m n
            y += tl.sum(x, 1)  # m

            x_ptrs += BLK_N

        y /= N
        y = tu.pow(y, 1 / p)

        y_ptrs = y_ptr + offs_m
        tl.store(y_ptrs, y, mask=offs_m < M)

    @staticmethod
    @triton.jit
    def backward_kernel(
        x_ptr, y_ptr, p, dy_ptr, dx_ptr, dp_ptr, eps, M, N, IS_P_TENSOR: tl.constexpr, BLK_M: tl.constexpr, BLK_N: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLK_M + tl.arange(0, BLK_M)  # m
        offs_n = tl.arange(0, BLK_N)  # n
        x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]  # m n
        dx_ptrs = dx_ptr + offs_m[:, None] * N + offs_n[None, :]  # m n
        mask_m = offs_m < M

        if IS_P_TENSOR:
            p = tl.load(p)

        # calculate y-level grad
        y = tl.load(y_ptr + offs_m, mask=mask_m)
        dy = tl.load(dy_ptr + offs_m, mask=mask_m)

        if IS_P_TENSOR:
            dp = tl.where(mask_m, -tl.log(y) / p * y * dy, 0.0)
            dp = tl.sum(dp, axis=0)
        dy = dy / p * y / (tu.pow(y, p) * N)  # m

        # re-calculate x for x-level grad
        for n in range(tl.cdiv(N, BLK_N)):
            mask = mask_m[:, None] & (offs_n[None, :] < N - n * BLK_N)
            x = tl.load(x_ptrs, mask=mask, other=0.0)  # m n

            # calculate adaptive average pooling
            x_ = tl.where((x < eps) & mask, eps, x)
            # x = tu.pow(x, p)
            # y += tl.sum(x, 1)  # m

            dx = tl.zeros((BLK_M, BLK_N), dtype=dy.dtype) + dy[:, None]  # m n

            if IS_P_TENSOR:
                dp_tmp = tl.where(mask, tu.pow(x_, p) * tl.log(x_) * dx, 0.0)
                dp += tl.sum(tl.sum(dp_tmp, axis=1), axis=0)

            dx *= p * tu.pow(x_, p - 1)
            dx = tl.where((x < eps) & mask, 0.0, dx)

            tl.store(dx_ptrs, dx, mask=mask)
            x_ptrs += BLK_N
            dx_ptrs += BLK_N

        if IS_P_TENSOR:
            tl.atomic_add(dp_ptr, dp)

    @staticmethod
    def forward(ctx, x: Tensor, p: Union[float, Tensor] = 3.0, eps: float = 1e-6, dim: int = -2, keepdim=True):
        ctx.is_p_tensor = isinstance(p, Tensor)
        dim = x.ndim + dim if dim < 0 else dim
        M = tu.cummul(*x.shape[:dim])
        N = tu.cummul(*x.shape[dim:])
        y = x.new_empty(list(x.shape[:dim]) + ([1 for _ in range(x.ndim - dim)] if keepdim else []))

        BLK_N = max(min(triton.next_power_of_2(N), 4096), 32)
        BLK_M = max(1, 1024 // BLK_N)
        grid = lambda meta: (triton.cdiv(M, meta["BLK_M"]),)
        GeMOps.forward_kernel[grid](x, y, p, eps, M, N, IS_P_TENSOR=ctx.is_p_tensor, BLK_M=BLK_M, BLK_N=BLK_N)

        if ctx.is_p_tensor:
            ctx.save_for_backward(x, p, y)
            ctx.params = eps, dim
        else:
            ctx.save_for_backward(x, y)
            ctx.params = p, eps, dim
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        if ctx.is_p_tensor:
            x, p, y = ctx.saved_tensors
            eps, dim = ctx.params
        else:
            x, y = ctx.saved_tensors
            p, eps, dim = ctx.params

        M = tu.cummul(*x.shape[:dim])
        N = tu.cummul(*x.shape[dim:])
        dx = th.empty_like(x)
        dp = None
        if ctx.is_p_tensor:
            dp = th.zeros_like(p)

        BLK_N = max(min(triton.next_power_of_2(N), 4096), 32)
        BLK_M = max(1, 1024 // BLK_N)
        grid = lambda meta: (triton.cdiv(M, meta["BLK_M"]),)
        GeMOps.backward_kernel[grid](x, y, p, dy, dx, dp, eps, M, N, IS_P_TENSOR=ctx.is_p_tensor, BLK_N=BLK_N, BLK_M=BLK_M)
        return dx, dp, None, None, None
