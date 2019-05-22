import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


class Identity(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def forward(self, x):
        return x


class AssertSize(nn.Module):
    """Assert that the input has the specified size.

    Example::

        net = nn.Sequential(
            tinder.nn.AssertSize(None, 3, 224, 224),
            nn.Conv2d(3, 64, kernel_size=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            tinder.nn.AssertSize(None, 128, 64, 64),
        )

    Args:
        size (iterable): an iterable of dimensions. Each dimension is one of -1, None, or positive integer.

    """

    def __init__(self, *size):
        super().__init__()
        self.size = [s if s != -1 else None for s in size]

    def __repr__(self):
        return f"AssertSize({self.size})"

    def forward(self, x):
        """
        """
        size = x.size()
        if len(self.size) != len(size):
            raise RuntimeError(
                f"expected rank {len(self.size)} but got a tensor of rank {len(size)}"
            )

        for expected, given in zip(self.size, size):
            if (expected is not None) and (expected is not given):
                raise RuntimeError(
                    f"expected size {self.size} but got a tensor of size {size}"
                )

        return x


def flatten(x):
    return x.view(x.size(0), -1)


class Flatten(nn.Module):
    """A layer that flattens the input.

    Example::

        net = nn.Sequential(
            nn.Conv2d(..),
            nn.BatchNorm2d(..),
            nn.ReLU(),

            nn.Conv2d(..),
            nn.BatchNorm2d(..),
            nn.ReLU(),

            tinder.nn.Flatten(),
            nn.Linear(3*3*512, 1024),
        )

    Args:
        x: input tensor
    """

    def forward(self, x):
        return flatten(x)


class View(nn.Module):
    """nn.Module version of tensor.view().

    Example::

        layer = tinder.nn.View(3, -1, 256)
        x = layer(x)

    The batch dimension is implicit.
    The above code is the same as `tensor.view(tensor.size(0), 3, -1, 256)`.

    Args:
        size_without_batch_dim (iterable): each dimension is one of -1, None, or positive.
    """

    def __init__(self, *size_without_batch_dim):
        super().__init__()
        self.size = [s if s != None else -1 for s in size_without_batch_dim]

    def forward(self, x):
        return x.view(x.size(0), *self.size)


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
        ws = tinder.nn.WeightScale(conv)
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

        std = (
            ((x - x.mean(dim=0, keepdim=True)).pow(2).mean(dim=0, keepdim=True) + 1e-8)
            .sqrt()
            .mean()
        )
        scalar = std.expand(x.size(0), 1, x.size(2), x.size(3))  # [N,1,H,W]
        x = torch.cat([x, scalar], dim=1)
        return x


def cross_entropy(input, target, *args, **kwargs):
    # If target label is out of range, the cross entropy is undefined.
    # This is to prevent crashes.
    target = target.clamp(0, input.shape[1] - 1)
    # target = torch.where(
    #     target >= 0 and target < input.shape[1],
    #     target,
    #     ?
    # )
    return torch.nn.functional.cross_entropy(input, target, *args, **kwargs)


def loss_wgan_gp(D, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Gradient Penalty for Wasserstein GAN.

    It generates random interpolations between the real and fake points,
    and tries to make grad norm close to 1.

    Note:
        - The second-order derivative is unstable in PyTorch.
          Some layers in your G/D may not work.
        - It assumes CUDA.

    Example::

        gp = tinder.nn.loss_wgan_gp(D, real_img, fake_img)
        loss = D(fake)-D(real)+10*gp

    Args:
        D (callable): A discriminator. Typically `nn.Module` or a lambda function that returns the score.
        real (torch.Tensor): real image batch.
        fake (torch.Tensor): fake image batch.

    Returns:
        (torch.Tensor): gradient penalty. optimizing on this loss will update D.
    """

    batch_size = real.size(0)
    e = real.new_empty((batch_size, 1, 1, 1)).uniform_()
    x_hat = (e * real + (1 - e) * fake).detach().requires_grad_(True)
    scores = D(x_hat)

    grads = torch.autograd.grad(
        scores,
        x_hat,
        grad_outputs=torch.ones_like(scores),
        retain_graph=True,
        create_graph=True,
    )[0]

    norms = grads.view(batch_size, -1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()



# def one_dimensional_euclidean_wasserstein_dist(x, y, p=2, weight_x=None, weight_y=None):
#     """1D wasserstein distance between p(X) and p(Y) where d(X,Y)=|X-Y|.

def one_dimensional_discrete_wasserstein_distance(px, py, p=2):
    """1D wasserstein distance between p(class) and p(class) where d(cls1,cls2)=(cls2!=cls2).

    Arguments:
        px {torch.Tensor} -- [N,D]. N discrete distributions of shape [D] where p(1)+..p(D)=1
        py {torch.Tensor} -- [N,D]

    Keyword Arguments:
        p {int} -- [description] (default: {2})
    """

    # === ((px-py).abs().sum(dim=1)/(2.0)).mean()
    return (px-py).abs().sum()/2.0/float(px.size(0))

def test_one_dimensional_discrete_wasserstein_distance():
    import pytest
    px = torch.Tensor([[0.1,0.2,0.3,0.4]])
    py = torch.Tensor([[0.3,0.1,0.1,0.5]])
    dist = one_dimensional_discrete_wasserstein_distance(px, py, p=2).item()
    assert pytest.approx(dist)==0.3**0.5

def sliced_wasserstein_distance(x, y, sample_cnt, p=2, weight_x=None, weight_y=None):
    """Calculated a stochastic sliced wasserstein distance between x and y.

    c(x,y) = ||x-y||p


    Arguments:
        x {torch.Tensor} -- A tensor of shape [N,*]. Samples from the distribution p(X)
        y {torch.Tensor} -- A tensor of shape [N,*]. Samples from the distribution p(Y)
        sample_cnt {int} -- A number of samples to estimate the distance
        p {int} -- L_p is used to calculate sliced w-dist
        weight_x {torch.Tensor} -- A tensor of shape [N] or None
        weight_y {torch.Tensor} -- A tensor of shape [N] or None

    Returns:
        scalar -- The sliced wasserstein distance (with gradient)
    """

    x = flatten(x)  # x: [N,D]
    y = flatten(y)  # y: [N,D]

    unit_vector = torch.randn(x.shape[1], sample_cnt, device=x.device)
    unit_vector = torch.nn.functional.normalize(  # each col has a norm 1
        unit_vector, p=2, dim=0
    )
    x = torch.matmul(x, unit_vector)  #  [N,D] * [D, samples] = [N,samples]
    y = torch.matmul(y, unit_vector)


    sorted_x, sort_index_x = x.sort(dim=0)  # [N,samples]
    sorted_y, sort_index_y = y.sort(dim=0)  # [N,samples]

    if (weight_x is None) and (weight_y is None):
        w_dist = (sorted_x-sorted_y).norm(p=p, dim=0).mean()
    elif weight_x is None:
        weight_y = torch.nn.functional.normalize(weight_y, p=1, dim=0)*weight_y.shape[0]
        weight_y = weight_y[sort_index_y]  # [N,samples]
        w_dist = (sorted_x-sorted_y*weight_y).norm(p=p, dim=0).mean()
    elif weight_y is None:
        weight_x = torch.nn.functional.normalize(weight_x, p=1, dim=0)*weight_x.shape[0]
        weight_x = weight_x[sort_index_x]  # [N,samples]
        w_dist = (sorted_x*weight_x-sorted_y).norm(p=p, dim=0).mean()
    else:
        weight_x = torch.nn.functional.normalize(weight_x, p=1, dim=0)*weight_x.shape[0]
        weight_x = weight_x[sort_index_x]  # [N,samples]
        weight_y = torch.nn.functional.normalize(weight_y, p=1, dim=0)*weight_y.shape[0]
        weight_y = weight_y[sort_index_y]  # [N,samples]
        w_dist = (sorted_x*weight_x-sorted_y*weight_y).norm(p=p, dim=0).mean()

    return w_dist.pow(1.0/p)


def odin(
    network: nn.Module, x: torch.Tensor, threshold, T=1000, epsilon=0.0012
) -> Tuple[bool, torch.Tensor]:
    """Decide if we should reject the prediction for the given example x using ODIN.

    Example::

        is_reject, max_p = tinder.nn.odin(resnet, imgs, threshold=0.003)
        # assert (is_reject == (max_p<0.003)).all()

    Arguments:
        network {nn.Module} -- A function returning logits
        x {torch.Tensor} -- input of [B,*]
        threshold {[type]} -- [description]

    Keyword Arguments:
        T {int} -- A parameter for temperature scailing (default: {1000})
        epsilon {float} -- A parameter for noisyness (default: {0.0012})

    Returns:
        Tuple[torch.Tensor, torch.Tensor] -- (is_reject of shape [B], max_p of shape [B])
    """

    x.requires_grad = True

    logits = network(x) / T
    predict = logits.argmax(dim=1)
    negative_log_softmax = F.cross_entropy(
        input=logits, target=predict, reduction="sum"
    )
    negative_grad = torch.autograd.grad(outputs=negative_log_softmax, inputs=x)[0]

    with torch.no_grad():
        x_noise = x - epsilon * negative_grad.sign()
        max_p, _idx = F.softmax(network(x_noise) / T, dim=1).max(dim=1)
        reject = max_p < threshold

    x.requires_grad = False
    return reject, max_p


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
