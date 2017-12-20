import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.size = [s if s != -1 else None for s in size]

    def forward(self, x):
        return x.view(x.size(0), *self.size)


class PixelwiseNormalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, eps=1e-8)


# improved wgan loss for D
# see https://arxiv.org/pdf/1704.00028.pdf
class GradientPenaltyLoss(nn.Module):
    def __init__(self, D, e):
        super().__init__()
        self.__dict__['D'] = D  # weak reference. D is not a submodule
        self.e = e
        self.x_hat = 0

    def forward(self, real, fake):
        # (grads*grads).mean().sqrt()
        x_hat = (self.e * real + (1 - self.e) * fake).detach()
        score = self.D(x_hat)
        grads = torch.autograd.grad(score, x_hat, retain_graph=True, create_graph=True)

        import pdb
        pdb.set_trace()

        return ((grads.norm(2, dim=1) - 1) ** 2).mean()
