import sys

def setup(*, logger_name='tinder', parse_args=False, trace=True, pdb_on_error=True):
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
        for token in sys.argv[1:]:
            if token[0] == '-':
                token = token[1:]
            if token[0] == '-':
                token = token[1:]
            idx = token.find('=')
            if idx == -1:
                continue
            else:
                os.environ[token[:idx]] = token[idx + 1:]

        if 'gpu' in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['gpu']


    if logger_name is not None:
        import logging
        import tqdm
        log = logging.getLogger(logger_name)
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

        return log
