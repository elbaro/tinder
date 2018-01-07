import torch
import torch.nn as nn
from torch.autograd import Variable


class WeightScale(nn.Module):
    """A weight normalizing layer used in PGGAN.

    How it works:

        1. Initialize your ConvLayer with weights from N(0, std=kaiming)
        2. WeightScale calculates the initial std of weights (from 1)
        3. Divide Conv.weights by std (from 2). Now Conv.weights are ~N(0,1)
        4. On forward, WeightScale multiply the input by std (from 2).
        5. Bias or Activation is applied after WeightScale. If Conv has bias, WeightScale steals it.

    Note that the scale factor calculated in 2 is constant permanently.

    Advantage:

        The author of PGGAN claims Adam is better at training weights in N(0,1) than training conv weights of different std.

    Example::

        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        ws = WeightScale(conv)
        nn.Sequential(
            conv,
            ws
        )


    Args:
        prev_layer: a layer (e.g. nn.Conv2d) with `weight` and optionally `bias`.
        init_with_leakiness: if given, it initializes `prev_layer` with `kaiming_normal`.
    """

    def __init__(self, prev_layer, init_with_leakiness: float = None):
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
    """Pixelwise Normalization used in PGGAN.
        It normalizes the input [B,C,H,W] so that the L2 norms over the C dimension is 1.
        There are B*H*W norms.

        Example::

            x = tinder.PixelwiseNormalize()(x)
    """

    def forward(self, x):
        denominator = (x.pow(2).mean(dim=1, keepdim=True) + 1e-8).sqrt()
        return x / denominator


# calculate std deviation for each feature in each location
# scalar = average over all feature & locations
# broadcast to constant feature map
class MinibatchStddev(nn.Module):
    """Layer for GAN Discriminator used in PGGAN.
        It penalizes when images in the minibatch look similar.
        For example, G generates fall into mode collapse and generate similar images,
        then stddev of the minibatch is small. D looks at the small stddev and thinks this is likely fake.

        This layer calculates stddev and provide it as additional channel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (N, C, H, W)

        Returns:
            (N, C+1, H, W): The last channel dimension is stddev.
        """

        std = ((x - x.mean(dim=0, keepdim=True)).pow(2).mean(dim=0, keepdim=True) + 1e-8).sqrt().mean()
        scalar = std.expand(x.size(0), 1, x.size(2), x.size(3))  # [N,1,H,W]
        x = torch.cat([x, scalar], dim=1)
        return x


def loss_wgan_gp(D, real: torch.cuda.FloatTensor, fake: torch.cuda.FloatTensor) -> torch.autograd.Variable:
    """Gradient Penalty for Wasserstein GAN.

    It generates random interpolations between the real and fake points,
    and tries to make grad norm close to 1.

    Note:
        - The second-order derivative is unstable in PyTorch.
          Some layers in your G/D may not work.
        - It assumes CUDA.

    Example::

        gp = tinder.wgan_gp(D, real_img, fake_img)
        loss = lambda*gp

    Args:
        D (callable): A discriminator. Typically `nn.Module` or a lambda function that returns the score.
        real (torch.cuda.FloatTensor): real sample. Note this is not a `Variable`.
        fake (torch.cuda.FloatTensor): fake sample. Note this is not a `Variable`.

    Returns:
        (Variable): gradient penalty.
    """

    batch_size = real.size(0)
    e = torch.cuda.FloatTensor(batch_size).uniform_().view(batch_size, 1, 1, 1)
    x_hat = Variable(e * real + (1 - e) * fake, requires_grad=True)
    scores = D(x_hat)
    grads = torch.autograd.grad(scores, x_hat, retain_graph=True, create_graph=True,
                                grad_outputs=torch.ones_like(scores)
                                )[0]

    norms = grads.view(batch_size, -1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()
