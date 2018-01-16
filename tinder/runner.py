def setup(*, trace=True, pdb_on_error=True, parse_args=True, logger=True, args=None):
    """
    Setup convinient utils to run a python script for deep learning.

    Examples:
          `./a.py train 4,5,6 lr=0.01`
          If a.py calls `tinder.setup()`, this is equivalent to `args=['train','4,5,6','lr=0.01]`.
          The following environments are set:

          - os.environ['mode'] = 'train'
          - os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'
          - os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
          - os.environ['lr'] = '0.01'

    Args:
        trace: use `backtrace` module to print stacktrace
        pdb_on_error: enter `pdb` mode when an exception is raised
        parse_args: parse `args` or `sys.args` of a form `script.py [mode] [gpus_comma_separated]`
        logger: setup the logging format
        args (list): if given, this is used instead of `sys.args`
    """
    import os

    if trace:
        import backtrace
        backtrace.hook(align=True)
    if pdb_on_error:
        import sys
        old_hook = sys.excepthook

        def new_hook(type, value, tb):
            old_hook(type, value, tb)
            if type != KeyboardInterrupt:
                import pdb
                pdb.post_mortem(tb)

        sys.excepthook = new_hook

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    if parse_args:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', help='cmd. e.g) train/val/test/preprocess', type=str)
        parser.add_argument('gpu', help='gpu list; e.g) 0,1', type=str)
        args, unknowns = parser.parse_known_args(args)

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        os.environ['mode'] = args.mode
        for unknown in unknowns:
            if unknown[0] == '-': unknown = unknown[1:]
            if unknown[0] == '-': unknown = unknown[1:]
            idx = unknown.find('=')
            if idx == -1:
                os.environ[unknown] = '1'
                print('잘못된 입력: ', unknown)
                sys.exit(-1)
            else:
                os.environ[unknown[:idx]] = unknown[idx + 1:]

    if logger:
        import logging, tqdm
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s [%(filename)16s:%(lineno)3s] %(message)s', datefmt='%m월%d일 %H시%M분')

        class TqdmLoggingHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.tqdm.write(msg)
                    self.flush()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    self.handleError(record)

        handler = TqdmLoggingHandler()
        handler.setFormatter(formatter)
        log.addHandler(handler)
