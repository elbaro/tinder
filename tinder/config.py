import sys
import os
from colorama import Fore, Style
from enum import Enum, auto
from types import SimpleNamespace


def bootstrap(*, logger_name="tinder", trace=True, pdb_on_error=True):
    """
    Setup convinient utils to run a python script for deep learning.

    Sets the 'CUDA_DEVICE_ORDER' environment variable to 'PCI_BUS_ID'

    Examples:
          `./a.py lr=0.01 gpu=0,1,2`

          The following environments are set:

          - os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
          - os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
          - os.environ['lr'] = '0.01'

    Args:
        logger_name: setup standard python logger that is compatiable with tqdm
        parse_args: set environment variables from command line arguments. gpu is alias for CUDA_DEVICE_DEVICES.
        trace: use `backtrace` module to print stacktrace
        pdb_on_error: enter `pdb` shell when an exception is raised
    """

    if trace:
        from .vendor import backtrace

        backtrace.hook(align=True)
    if pdb_on_error:
        old_hook = sys.excepthook

        def new_hook(type_, value, tb):
            old_hook(type_, value, tb)
            if type_ != KeyboardInterrupt:
                import pdb

                pdb.post_mortem(tb)

        sys.excepthook = new_hook

    if logger_name is not None:
        import logging
        import tqdm

        log = logging.getLogger(logger_name)
        log.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(filename)16s:%(lineno)3s] %(message)s",
            datefmt="%m/%d %H:%M",
        )

        class TqdmLoggingHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.tqdm.write(msg)
                    self.flush()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:  # pylint: disable=W0703
                    self.handleError(record)

        handler = TqdmLoggingHandler()
        handler.setFormatter(formatter)
        log.addHandler(handler)

        return log


class Placeholder(Enum):
    """A placeholder to denote the required parameter without default.

    Example::

        config = {
            'lr': 0.01,
            'gpu': Placeholder.STR
        }
        tinder.config.override(config)
    """

    INT = auto()
    FLOAT = auto()
    STR = auto()
    BOOL = auto()


def override(config):
    """Update your config dict with command line arguments.

    The original dictionary is the default values.
    To prevent mistakes, any command line argument not specified in the dictionary raises.

    `gpu` is a special key. If `gpu` is found,
    `os.environ['CUDA_VISIBLE_DEVICES']` is set accordingly.

    Example::

        config = {
            'lr': 0.01,
            'gpu': Placeholder.STR
        }
        tinder.config.override(config)

        // shell
        python script.py lr=0.005 gpu=0,1,2


    Args:
        config (dict or SimpleNamspace): parameter specs

    Raises:
        RuntimeError: the command line provided an unknown arg.
        RuntimeError: the required arg is not provided.
    """

    if isinstance(config, SimpleNamespace):
        style = "simple"
        config = config.__dict__
    else:
        style = "dict"

    new = {}

    for token in sys.argv[1:]:
        idx = token.find("=")
        if idx == -1:
            continue
        else:
            key = token[:idx]
            value = token[idx + 1 :]

            if key not in config:
                raise RuntimeError("unknown arg: " + key)

            default = config[key]
            if isinstance(default, bool) or default == Placeholder.BOOL:
                value = value == "True"  # bool('False')==True
            elif isinstance(default, int) or default == Placeholder.INT:
                value = int(value)
            elif isinstance(default, float) or default == Placeholder.FLOAT:
                value = float(value)
            elif isinstance(default, str) or default == Placeholder.STR:
                pass

            new[key] = value
            config[key] = value

    print(f"{Fore.YELLOW}=========={Style.RESET_ALL}")
    if config:
        width = max(len(key) for key in config.keys())
        for key, value in config.items():
            if key in new:
                print(f"{Fore.GREEN}{key:>{width}s}: {value}{Style.RESET_ALL}")
            elif type(value) == Placeholder:
                raise RuntimeError(f"Required: {key}")
            else:
                print(f"{key:>{width}s}: {value}")
    else:
        print("no config")
    print(f"{Fore.YELLOW}=========={Style.RESET_ALL}")

    # bonus
    if "gpu" in config:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

    if style == "simple":
        new = SimpleNamespace(**new)

    return new
