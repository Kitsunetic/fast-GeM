import triton
import triton.language as tl


@triton.jit
def pow(x, p):
    return tl.exp(tl.log(x) * p)


def cummul(*x):
    y = 1
    for v in x:
        y *= v
    return y
