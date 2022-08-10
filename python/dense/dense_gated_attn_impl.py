# %%
import torch
from torch.autograd import Function
import triton
import triton.language as tl


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
    indices = indices.repeat(grad.shape[-2], 1) - indices[:grad.shape[-2], None]
    indices = indices - eps[:, 1:2]
    w = 1 / (1 + (beta * (indices - eps[:, 0:1])).exp() + (-beta * (indices + eps[:, 0:1])).exp())
    
    dy_de = torch.empty_like(eps)
    dy_de[:, 0] = (beta * (1 - w) * dy_dx).sum(-1)
    dy_de[:, 1] = (-beta * (w - 1) * torch.tanh(beta * indices) * dy_dx).sum(-1)
    return dy_dx, dy_de


# %%
@triton.jit
def soft_gate(x, a, b):
    return 1 / (1 + tl.exp(b * (x - a)) + tl.exp(-b * (x + a)))

@triton.jit
def sigmoid(x):
        return 1 / (1 + tl.exp(-x))

@triton.jit
def tanh(x):
    return 2 * sigmoid(2 * x) - 1 


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
def d_gated_softmax_kernel_v1(
        grad_ptr,               # [..., L, L] 
        y_ptr,                  # [..., L, L]
        eps_ptr,                # [..., L, 2]
        dx_ptr,                 # [..., L, L]
        de_ptr,                 # [..., L, 2]
        beta,                   # [1]
        L,                      # int
        BLOCK_L: tl.constexpr,  # int 
        causal: tl.constexpr    # bool
    ):
    idr = tl.program_id(0)
    
    block = tl.arange(0, BLOCK_L)
    if causal:  # apply casual triangular masking
        attn_mask = block <= (idr % L)
        store_mask = block < L  # make separate store mask in case dy_ptr is not initialized with zeros
    else:
        attn_mask = block < L
        store_mask = attn_mask
    row = block + idr * L

    grad = tl.load(grad_ptr + row, attn_mask, other=0)
    Y = tl.load(y_ptr + row, attn_mask, other=0)
    g_sum = tl.sum(grad * Y, 0)
    # basically the softmax derivative
    dy_dx = Y * (grad - g_sum)
    tl.store(dx_ptr + row, dy_dx, store_mask)

    alpha = tl.load(eps_ptr + idr * 2)
    shift = tl.load(eps_ptr + idr * 2 + 1)
    beta_ = tl.load(beta)
    center = idr.to(tl.float32)
    indices = block.to(tl.float32) - (center + shift)

    w = soft_gate(indices, alpha, beta_)
    dy_da = beta * (1 - w) * dy_dx

    tl.store(de_ptr + idr * 2, tl.sum(dy_da, 0))

    dy_ds = tanh(beta * indices) * dy_da
    tl.store(de_ptr + idr * 2 + 1, tl.sum(dy_ds, 0))


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

