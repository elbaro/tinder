import torch
import torch.nn as nn
from torch.autograd import Variable


# improved wgan loss for D
# see https://arxiv.org/pdf/1704.00028.pdf

# How to use:
# prev_layer is initialized with He init ~ N(0, sqrt(2/fan_in..))
# __init__ -> Make prev_layer.weight's std to 1 && set bias to 0
# forward -> Make std back to He sqrt(2/fan_in..).
class WeightScale(nn.Module):
    def __init__(self, prev_layer, init_with_leakiness=None):
        super().__init__()

        if init_with_leakiness is not None:
            torch.nn.init.kaiming_normal(prev_layer.weight, a=init_with_leakiness)

        self.scale = torch.nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=False)
        self.scale.data[0] = torch.mean(prev_layer.weight.data ** 2) ** 0.5 + 1e-8
        prev_layer.weight.data.div_(self.scale.data[0])

        if prev_layer.bias is not None:
            self.bias = prev_layer.bias
            self.bias.data.zero_()
            prev_layer.bias = None
        else:
            self.bias = None

    def forward(self, x):
        x = self.scale * x
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size(0), 1, 1)
        return x

    def __repr__(self):
        # param_str = '(prev_layer = %s)' % (self.prev_layer.__class__.__name__)
        return self.__class__.__name__


class PixelwiseNormalize(nn.Module):
    def forward(self, x):
        denominator = (x.pow(2).mean(dim=1, keepdim=True) + 1e-8).sqrt()
        return x / denominator


# calculate std deviation for each feature in each location
# scalar = average over all feature & locations
# broadcast to constant feature map
class MinibatchStddev(nn.Module):
    def __init__(self):
        super().__init__()

    # x : [N, 512, H, W]
    # out:[N, 513, H, W]
    def forward(self, x):
        # std = x.std(dim=0)
        # batch_size = x.size(0)
        std = ((x - x.mean(dim=0, keepdim=True)).pow(2).mean(dim=0, keepdim=True) + 1e-8).sqrt().mean()
        # std = (x - x.mean(dim=0, keepdim=True)).norm(p=2, dim=0).mean() / (
        #     (batch_size - 1) ** 0.5)  # std = [C,H,W] -> [1]
        scalar = std.expand(x.size(0), 1, x.size(2), x.size(3))  # [N,1,H,W]
        x = torch.cat([x, scalar], dim=1)
        return x


def loss_wgan_gp(D, real_img: torch.cuda.FloatTensor, fake_img: torch.cuda.FloatTensor):
    batch_size = real_img.size(0)
    e = torch.cuda.FloatTensor(batch_size).uniform_().view(batch_size, 1, 1, 1)
    x_hat = Variable(e * real_img + (1 - e) * fake_img, requires_grad=True)
    scores = D(x_hat)
    grads = torch.autograd.grad(scores, x_hat, retain_graph=True, create_graph=True,
                                grad_outputs=torch.ones_like(scores)
                                )[0]

    norms = grads.view(batch_size, -1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()
