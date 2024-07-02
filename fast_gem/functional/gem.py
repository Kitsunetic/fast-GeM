from typing import Union

import torch as th
from torch import Tensor

from .gem_torch import gem_torch_1d, gem_torch_2d, gem_torch_3d
from .gem1d import gem_ops1d
from .gem2d import gem_ops2d

# from .gem3d import gem_ops3d

__all__ = ["gem"]


_fn = {
    (3, False): gem_torch_1d,
    (4, False): gem_torch_2d,
    (5, False): gem_torch_3d,
    (3, True): gem_ops1d,
    (4, True): gem_ops2d,
    (5, True): gem_torch_3d,
}


def gem(x: Tensor, p: Union[float, Tensor] = 3.0, eps: float = 1e-6, dim: int = -2, keepdim=True, fused=True):
    assert x.device == th.device("cpu") ^ fused, "Cannot use fused operation with CPU tensors"
    assert (x.ndim, fused) in _fn, f"Unknown input shape: {x.shape}"

    return _fn[(x.ndim, fused)](x, p, eps, keepdim=keepdim)
