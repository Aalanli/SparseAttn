# %%
import functools
import math
import torch
from bench.utils import build_extension, bench

ext = build_extension('matmul', 'src/attn_impl.cu')

def attn_cuda(q, k, v, block_div=1, halt=-1):
    return ext.attn(q, k, v, block_div, halt)

def attn(q, k, v):
    a1 = q @ k.transpose(-1, -2)
    a1 /= math.sqrt(q.shape[-1])
    a1: torch.Tensor
    rmax = torch.max(a1, -1).values
    rsum = a1.sum(-1)
    a1 = a1.softmax(-1)
    y = a1 @ v
    return y, rmax, rsum

def simple_attn(q, k, v):
    a1 = q @ k.t()
    return a1 @ v

q = torch.rand(2, 1024, 512, device='cuda')
k = torch.rand_like(q)
v = torch.rand_like(q)

y1 = attn(q, k, v)
y = attn_cuda(q, k, v)

for i in range(len(y)):
    print(torch.allclose(y[i], y1[i]))

# %%
constructor = lambda N: (torch.rand(N, 512, device='cuda'), torch.rand_like(q), torch.rand_like(q))

halt = -1
attn_t2 = functools.partial(attn_cuda, halt=halt, block_div=1)
attn_t2.__name__ = 'attnv2_t'


bench([attn_t2, attn], constructor, 16, 8)

