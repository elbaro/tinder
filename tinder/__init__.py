"""
Tinder

A library complement to PyTorch.
"""

from .image import BoundingBox, crop
from .config import bootstrap, Placeholder, override
from .optimizer import WarmRestartLR, copy_opt_state
from .saver import Saver
from .visualize import show_imgs
from .metrics import instance_segmentation_iou, semantic_segmentation_iou
from .nn import WeightScale, PixelwiseNormalize, MinibatchStddev, loss_wgan_gp
from .layers import AssertSize, Flatten, View
from .dataset import hash_group, DataLoaderIterator, BalancedDataLoader
from .monitor import Stat, Stats
from .queue import RedisQueue, RabbitConsumer, RabbitProducer
