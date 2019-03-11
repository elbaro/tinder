"""
Tinder

A library complement to PyTorch.
"""

from . import nn
from . import image
from . import config
from . import optimizer
from . import saver
from . import visualize
from . import metrics
from . import dataset
from . import monitor
from . import queue
from . import rl
from . import model

# shortcuts
from .config import bootstrap, Placeholder, override
from .monitor import Stat, Stats
from .saver import Saver
