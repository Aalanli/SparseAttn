# %%
import torch
from torch.autograd import Function
import triton
import triton.language as tl

from gated_attn import d_gated_dense_softmax_cuda

# %%
def gated_softmax_torch(x, eps, beta):
    indices = torch.arange(0, x.shape[-1], 1, dtype=x.dtype, device=x.device)
    indices = indices.repeat(x.shape[-1], 1) - indices[:, None]
    indices = indices - eps[:, 1:2]
    w = 1 / (1 + (beta * (indices - eps[:, 0:1])).exp() + (-beta * (indices + eps[:, 0:1])).exp())
    y = (x - x.max(-1).values[:, None]).exp() * w
    y = y / (y.sum(-1)[:, None])
    return y


def d_gated_softmax_torch(grad, y, eps, beta):
    g_sum = (grad * y).sum(-1)
    dy_dx = y * (grad - g_sum[:, None])

    indices = torch.arange(0, grad.shape[-1], 1, dtype=y.dtype, device=y.device)
    indices = indices.repeat(grad.shape[-1], 1) - indices[:, None]
    indices = indices - eps[:, 1:2]
    w = 1 / (1 + (beta * (indices - eps[:, 0:1])).exp() + (-beta * (indices + eps[:, 0:1])).exp())
    
    dy_de = torch.empty_like(eps)
    dy_de[:, 0] = (beta * (1 - w) * dy_dx).sum(-1)
    dy_de[:, 1] = (-beta * (w - 1) * torch.tanh(beta * indices) * dy_dx).sum(-1)
    return dy_dx, dy_de
    

grad = torch.rand(512, 512, device='cuda')
y = torch.rand_like(grad)
eps = torch.rand(512, 2, device='cuda')
beta = 2.0

dydx1, dyde1 = d_gated_softmax_torch(grad, y, eps, beta)
dydx2, dyde2 = d_gated_dense_softmax_cuda(grad, y, eps, beta, 2)

print(torch.allclose(dydx1, dydx2))
print(torch.allclose(dyde1, dyde2))



# %%
@triton.jit
def soft_gate(x, a, b):
    return 1 / (1 + tl.exp(b * (x - a)) + tl.exp(-b * (x + a)))


@triton.jit
def gated_softmax_kernel_v1(x_ptr, eps_ptr, y_ptr, beta, D, BLOCK_D: tl.constexpr, causal: tl.constexpr):
    idr = tl.program_id(0)
    
    block = tl.arange(0, BLOCK_D)
    mask = block < D
    if causal:
        causal_mask = block <= (idr % D)
    else:
        causal_mask = mask
    row = block + idr * D

    X = tl.load(x_ptr + row, causal_mask, -float('inf'))
    alpha = tl.load(eps_ptr + idr * 2)
    shift = tl.load(eps_ptr + idr * 2 + 1)
    beta_ = tl.load(beta)
    center = idr.to(tl.float32)

    maximum = tl.max(X, 0)
    X = tl.exp(X - maximum)
    X *= soft_gate(block.to(tl.float32) - (center + shift), alpha, beta_)
    sum = tl.sum(X, 0)

    Y = X / sum
    tl.store(y_ptr + row, Y, mask)


def gated_softmax_v1(X: torch.Tensor, epsilion: torch.Tensor, beta, causal=False):
    Y = torch.empty_like(X)
    cols = X.shape[-1]
    rows = epsilion.nelement() // 2
    gated_softmax_kernel_v1[(rows,)](
        X, epsilion, Y, beta, cols, triton.next_power_of_2(cols), causal
    )
    return Y


# triton does not support tanh, so backward gradients are not as accurate as they can be
@triton.jit
def d_gated_softmax_kernel_v1(grad_ptr, y_ptr, eps_ptr, dx_ptr, de_ptr, beta, D, BLOCK_D: tl.constexpr, causal: tl.constexpr):
    idr = tl.program_id(0)
    
    block = tl.arange(0, BLOCK_D)
    if causal:
        causal_mask = block <= (idr % D)
    else:
        causal_mask = block < D
        mask = causal_mask
    row = block + idr * D

    grad = tl.load(grad_ptr + row, causal_mask, 0)
    Y = tl.load(y_ptr + row, causal_mask, 0)
    g_sum = tl.sum(grad * Y, 0)
    dy_dx = Y * (grad - g_sum)
    tl.store(dx_ptr + row, dy_dx, mask)

    alpha = tl.load(eps_ptr + idr * 2)
    shift = tl.load(eps_ptr + idr * 2 + 1)
    beta_ = tl.load(beta)
    center = idr.to(tl.float32)
    indices = block.to(tl.float32) - (center + shift)

    e1 = tl.exp(beta_ * indices)
    e2 = tl.exp(beta_ * alpha)

    w = 1 - (1 / (1 + (1 / e2) * (e1 + (1 / e1))))
    h = 2 / (1 + e1 * (e1 + e2))

    dy_da = beta_ * tl.sum(dy_dx * w, 0)
    dy_ds = dy_da - beta_ * tl.sum(dy_dx * h, 0)

    tl.store(de_ptr + idr * 2, dy_da)
    tl.store(de_ptr + idr * 2 + 1, dy_ds)


def d_gated_softmax_v1(grad, Y: torch.Tensor, epsilion: torch.Tensor, beta, causal=False):
    dx = torch.empty_like(Y)
    de = torch.empty_like(epsilion)
    cols = Y.shape[-1]
    rows = epsilion.nelement() // 2
    d_gated_softmax_kernel_v1[(rows,)](
        grad, Y, epsilion, dx, de, beta, cols, triton.next_power_of_2(cols), causal
    )
    return dx, de


class GatedSoftmax(Function):
    @staticmethod
    def forward(ctx, x, epsilion, beta, causal=False):
        #y = gated_softmax_v1(x, epsilion, beta, causal)
        y = gated_softmax_v1(x, epsilion, beta)
        ctx.save_for_backward(y, epsilion, beta)
        ctx.causal = causal
        return y
    
    @staticmethod
    def backward(ctx, grad, *args):
        y, eps, beta = ctx.saved_tensors
        #dx, de = d_gated_softmax_v1(grad, y, eps, beta, ctx.causal)
        dx, de = d_gated_softmax_v1(grad, y, eps, beta)
        return dx, de, None, None

