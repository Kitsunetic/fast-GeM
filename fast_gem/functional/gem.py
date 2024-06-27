from typing import Union

import torch as th
from torch import Tensor

from .gem_torch import gem_torch

__all__ = ["gem"]


_triton_available = True


def gem(x: Tensor, p: Union[float, Tensor] = 3.0, eps: float = 1e-6, dim: int = -2, keepdim=True):
    global _triton_available

    if x.device == th.device("cpu"):
        return gem_torch(x, p, eps, dim, keepdim)

    try:
        assert _triton_available
        from .gem_triton import GeMOps

        return GeMOps.apply(x, p, eps, dim, keepdim)
    except:
        _triton_available = False
        return gem_torch(x, p, eps, dim, keepdim)
