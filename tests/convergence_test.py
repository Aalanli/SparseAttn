# %%
import math
import torch

from matplotlib import pyplot as plt

def softgate(x, s, a, b):
    return 1 / (1 + (b * (x - s - a)).exp() + (-b * (x - s + a)).exp())

def gaussian(x, mean):
    return (1 / math.sqrt(2 * math.pi)) * ((x - mean) ** 2 / -2).exp()

x = torch.arange(-10, 10, 0.1)

g = gaussian(x, 0)
plt.plot(x, g)
plt.show()

s = torch.tensor(-3.0, requires_grad=True)
a = torch.tensor(3.0, requires_grad=True)
k = softgate(x, 4, 3, 0.1)
plt.plot(x, k)
plt.show()
# %%
xs = torch.linspace(-50, 100, 500)
ys = torch.linspace(-50, 100, 500)

gx, gy = torch.meshgrid(xs, ys, indexing='xy')
z = (g[None, None, :] - softgate(x[None, None, :], gx[:, :, None], gy[:, :, None], 0.1)) ** 2
z = z.sum(-1)
ax = plt.axes(projection='3d')
ax.plot_surface(gx.numpy(), gy.numpy(), z.numpy())
plt.show()
