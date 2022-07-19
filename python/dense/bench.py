# %%
import torch
from python.dense.dense_gated_attn_impl import d_gated_softmax_v1, d_gated_softmax_torch
from python.bench_utils import bench
import gated_attn
from gated_attn import d_gated_dense_softmax_cuda
import triton

grad = torch.rand(32, 32, device='cuda', dtype=torch.float)
y = torch.rand_like(grad)
eps = torch.rand(32, 2, device='cuda', dtype=torch.float)
eps *= 0
eps += 2
beta = 2.0

dydx1, dyde1 = d_gated_softmax_torch(grad, y, eps, beta)
dydx2, dyde2 = d_gated_dense_softmax_cuda(grad, y, eps, beta, 1)

print(torch.allclose(dydx1, dydx2))
print(torch.allclose(dyde1, dyde2))
print((dyde1 - dyde2).abs().max())

# %%
constructor = lambda N: (
    torch.rand(512, N, device='cuda', dtype=torch.float),
    torch.rand(512, N, device='cuda', dtype=torch.float),
    torch.rand(512, 2, device='cuda', dtype=torch.float),
    2
)
kernel1 = lambda g, y, eps, beta: d_gated_dense_softmax_cuda(g, y, eps, beta, 0)
kernel1.__name__ = "kernel1"
kernel2 = lambda g, y, eps, beta: d_gated_dense_softmax_cuda(g, y, eps, beta, 1)
kernel2.__name__ = "kernel2"
kernel3 = lambda g, y, eps, beta: d_gated_softmax_v1(g, y, eps, torch.tensor([2.0], dtype=torch.float, device='cuda'))
kernel3.__name__ = "triton-kernel"
bench([kernel1, kernel2, kernel3], constructor, range(128, 1024 * 4, 512))

# %%
constructor = lambda N: (
    torch.rand(512, N, device='cuda', dtype=torch.float),)
softmax_torch = lambda x: torch.softmax(x, -1)
softmax_torch.__name__ = "softmax_torch"
bench([gated_attn.softmax_ref, softmax_torch], constructor, range(128, 1024 * 4, 512))