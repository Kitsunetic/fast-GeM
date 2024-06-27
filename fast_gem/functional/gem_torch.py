import torch as th
import torch.nn.functional as F
from kitsu.utils import cummul
from torch import Tensor


def gem_torch_old(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def gem_torch(x: Tensor, p=3, eps=1e-6, dim=-2, keepdim=True):
    """
    input:
    - x: ... ... (= m n)
    return:
    - y: ... (= m)
    """
    dim = x.ndim + dim if dim < 0 else dim
    yshape = list(x.shape[:dim]) + ([1 for _ in range(x.ndim - dim)] if keepdim else [])

    M = cummul(*x.shape[:dim])
    N = cummul(*x.shape[dim:])
    x = x.view(M, N)  # m n

    x = x.clamp(eps).pow_(p)
    x = x.mean(-1)  # adaptive average pooling
    x = x.pow_(1.0 / p)

    return x.view(*yshape)
