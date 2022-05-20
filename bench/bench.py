# %%
import functools
import math
import torch
from bench.utils import build_extension, bench

ext = build_extension('matmul', 'src/attn_impl.cu')

def attnv1_t(q, k, v):
    return ext.attnv1_t(q, k, v)

def attnv2_t(q, k, v, halt=-1, block_div=1):
    return ext.attnv2_t(q, k, v, block_div, halt)

def attnv3_t(q, k, v, halt=-1, block_div=1):
    return ext.attnv3_t(q, k, v, block_div, halt)

def attn(q, k, v):
    a1 = q @ k.t()
    a1 /= math.sqrt(q.shape[-1])
    a1: torch.Tensor
    rmax = torch.max(a1, -1).values
    rsum = a1.sum(-1)
    a1 = a1.softmax(-1)
    y = a1 @ v
    return y

def simple_attn(q, k, v):
    a1 = q @ k.t()
    return a1 @ v

q = torch.rand(1024, 512, device='cuda')
k = torch.rand_like(q)
v = torch.rand_like(q)

y = attnv3_t(q, k, v)
y1 = attn(q, k, v)

torch.allclose(y, y1)

# %%
constructor = lambda N: (torch.rand(N, 512, device='cuda'), torch.rand_like(q), torch.rand_like(q))

halt = 64
attn_t2 = functools.partial(attnv2_t, halt=halt, block_div=1)
attn_t2.__name__ = 'attnv2_t'

attn_t3 = functools.partial(attnv3_t, halt=halt, block_div=1)
attn_t3.__name__ = 'attnv3_t'

bench([attn_t2, attn_t3], constructor, 16, 8)

