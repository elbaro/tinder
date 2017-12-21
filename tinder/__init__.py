import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AssertSize(nn.Module):
    # size is a list of dimensions.
    # dimension is a positive number, -1, or None
    def __init__(self, *size):
        super().__init__()
        self.size = [s if s != -1 else None for s in size]

    def __repr__(self):
        return f'AssertSize({self.size})'

    def forward(self, x):
        size = x.size()
        if len(self.size) != len(size):
            raise RuntimeError(f"expected rank {len(self.size)} but got a tensor of rank {len(size)}")

        for expected, given in zip(self.size, size):
            if expected != None and expected != given:
                raise RuntimeError(f"expected size {self.size} but got a tensor of size {size}")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class View(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size
        self.size = [s if s != None else -1 for s in size]

    def forward(self, x):
        return x.view(x.size(0), *self.size)


class PixelwiseNormalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, eps=1e-8)


# improved wgan loss for D
# see https://arxiv.org/pdf/1704.00028.pdf

class WScaleLayer(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x


def wgan_gp(D, real_img, fake_img):
    # (grads*grads).mean().sqrt()
    batch_size = real_img.size(0)
    e = Variable(torch.cuda.FloatTensor(batch_size).random_().view(batch_size, 1, 1, 1))
    x_hat = (e * real_img + (1 - e) * fake_img).detach()
    score = D(x_hat)
    grads = torch.autograd.grad(score, x_hat, retain_graph=True, create_graph=True)
    norms = grads.view(batch_size, -1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()


def DataLoaderIterator(loader, num=None, last_step=0):
    step = last_step
    while True:
        for batch in loader:
            step += 1
            if step > num:
                return
            yield step, batch
