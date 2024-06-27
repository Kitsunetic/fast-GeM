from typing import Union

import torch as th
import torch.nn as nn
from torch import Tensor

from .functional import gem


class GeM(nn.Module):
    def __init__(
        self,
        p: Union[float, int, Tensor] = 3.0,
        eps: float = 1e-6,
        p_trainable: bool = True,
        dim: int = -2,
        keepdim=True,
    ):
        """
        Generalized Mean (GeM) Pooling Layer.

        This layer applies the Generalized Mean (GeM) pooling operation over an input tensor.
        GeM is a generalization of average and max pooling, controlled by the parameter `p`.

        Args:
        - p (float | int | Tensor): The pooling parameter. It can be trainable by using `p_trainable=True` (default=True).
        - eps (float): A small value added to the input tensor to avoid numerical instability.
        - p_trainable (bool): If True, `p` is a learnable parameter. Otherwise, `p` is fixed.
        - dim (int): The dimension over which the pooling operation is applied.
            Default is -2, this will reduce last 2 dimensions so usually useful with image tensor.
            For 1D tensor like audio and 4D tensor like voxel, you can use dim=-1 and dim=-3 respectively.
        """
        super().__init__()

        if p_trainable:
            if isinstance(p, Tensor):
                self.p = nn.Parameter(p.to(th.float))
            elif isinstance(p, (int, float)):
                self.p = nn.Parameter(th.tensor(p, dtype=th.float))
            else:
                raise NotImplementedError(f"Unknown data type for `p`: {type(p)}")
        else:
            self.p = p

        self.eps = eps
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor, dim=-2):
        return gem(x, self.p, self.eps, dim=dim, keepdim=self.keepdim)

    def __repr__(self):
        p_val = self.p
        if isinstance(p_val, Tensor):
            p_val = p_val.item()

        return f"{self.__class__.__name__}(p={p_val:.4f}), eps={self.eps})"
