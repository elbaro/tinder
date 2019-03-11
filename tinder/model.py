from types import SimpleNamespace
import tinder
import torch
import tqdm
import tensorboardX
from colorama import Fore, Style


class Model(object):
    OPT_MAP = {
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
        'adam': torch.optim.Adam,
    }

    def __init__(self,
                 name,
                 net,
                 opt=None,
                 log_dir=None,
                 weight_dir=None,
                 load='latest'):

        self.net = net
        if isinstance(opt, torch.optim.Optimizer):
            self.opt = opt
        elif isinstance(opt, dict):
            self.opt = Model.OPT_MAP[opt['name']](*opt['args'],
                                                  **opt['kwargs'])

        self.bundle = SimpleNamespace(
            net=net,
            opt=self.opt,
            epoch=0,
            step=0,
        )

        if log_dir is not None:
            self.tb = tensorboardX.SummaryWriter(log_dir)
        else:
            self.tb = None

        if weight_dir is not None:
            self.saver = tinder.saver.Saver(weight_dir, name)

            if load == 'latest':
                self.saver.load_latest(self.bundle)
            elif load == 'best':
                self.saver.load_best(self.bundle)
            elif isinstance(load, int):
                if not self.saver.load(self.bundle, epoch=load):
                    raise RuntimeError('fail to load epoch=', load)
            else:
                raise RuntimeError('cannot understand load arg', load)
        else:
            self.saver = None

    def train_epoch(self,
                    loader,
                    train_minibatch_fn,
                    is_logging=True,
                    manual_opt=False):
        assert (self.opt is not None) or manual_opt
        if is_logging and (self.tb is None):
            assert 'log_dir is not given'

        self.net.train()

        loader = tqdm.tqdm(loader, leave=False)
        step = self.bundle.step

        n = 0
        metrics = {}
        for batch in loader:
            assert isinstance(batch, dict)
            step += 1

            batch_size = batch['batch_size'] = len(batch)
            n += batch_size

            result = train_minibatch_fn(batch)

            if not manual_opt:
                self.opt.zero_grad()
                result['loss'].backward()
                self.opt.step()

            if is_logging:
                result['loss'] = result['loss'].item()
                msg = "[train]"
                for (k, v) in result.items():
                    if torch.is_tensor(v):
                        v = v.item()

                    self.tb.add_scalar(k, v, step)
                    msg += f"    {k} = {v:8.4}"
                    v *= batch_size
                    if k in metrics:
                        metrics[k] += v
                    else:
                        metrics[k] = v
                loader.set_description(msg)

        self.bundle.step = step
        self.bundle.epoch += 1

        # {
        #   'loss': ~,
        #   'acc': ~,
        # }
        for k in metrics:
            metrics[k] /= n
        return metrics

    def eval(self, loader, eval_minibatch_fn):
        self.net.eval()

        for batch in loader:
            if isinstance(batch, dict):
                batch['batch_size'] = len(batch)
            log = eval_minibatch_fn(batch)

    def train(self,
              epochs,
              train_loader,
              train_minibatch_fn,
              logging='step',
              save_epoch_interval=1,
              score_col=None,
              manual_opt=False,
              eval_loader=None,
              eval_minibatch_fn=None):

        for epoch in tqdm.trange(self.bundle.epoch + 1, epochs + 1):
            metrics = self.train_epoch(train_loader, train_minibatch_fn,
                                       logging == 'step', manual_opt)

            train_msg = ""
            for (k, v) in metrics.items():
                train_msg += f"{k} = {v:8.4}    "

            if (eval_loader is not None) and (eval_minibatch_fn is not None):
                self.eval(eval_loader, eval_minibatch_fn)
                eval_msg = ""
                for (k, v) in metrics.items():
                    eval_msg += f"{k} = {v:8.4}    "

                tqdm.tqdm.write(
                    f'epoch {epoch:3d} [train] {Fore.YELLOW}{train_msg}{Style.RESET_ALL}'
                    + f'    [valid] {Fore.GREEN}{eval_msg}{Style.RESET_ALL}')
            else:
                tqdm.tqdm.write(
                    f'epoch {epoch:3d} [train] {Fore.YELLOW}{train_msg}{Style.RESET_ALL}'
                )

            if (self.saver is not None) and (epoch % save_epoch_interval) == 0:
                self.saver.save(
                    self.bundle,
                    epoch=self.bundle.epoch,
                    score=metrics['score_col'] if
                    (score_col is not None) else 0.0)

    def infer(self, net, loader, infer_minibatch_fn, id_col='id'):
        net.eval()
        result = {}

        for batch in loader:
            assert isinstance(batch, dict)
            assert 'id' in batch

            batch['batch_size'] = len(batch)
            minibatch_result = infer_minibatch_fn(batch)
            assert len(minibatch_result) == len(batch)
            for res in zip(batch[id_col], minibatch_result):
                result[res[0]] = res[1]

        return result
