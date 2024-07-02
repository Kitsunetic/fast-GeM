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
        self.keepdim = keepdim

    def forward(self, x: Tensor):
        return gem(x, self.p, self.eps, keepdim=self.keepdim)

    def __repr__(self):
        p_val = self.p
        if isinstance(p_val, Tensor):
            p_val = p_val.item()

        return f"{self.__class__.__name__}(p={p_val:.4f}), eps={self.eps})"
