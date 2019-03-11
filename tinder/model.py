from types import SimpleNamespace
import tinder
import torch
import tqdm
from colorama import Fore, Style


class Model(object):
    OPT_MAP = {
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adam": torch.optim.Adam,
    }

    def __init__(
        self, name, net, opt=None, logger_fn=None, weight_dir=None, load="latest"
    ):

        self.net = net
        if isinstance(opt, torch.optim.Optimizer):
            self.opt = opt
        elif isinstance(opt, dict):
            self.opt = Model.OPT_MAP[opt["name"]](*opt["args"], **opt["kwargs"])

        self.bundle = SimpleNamespace(net=net, opt=self.opt, epoch=0, step=0)

        self.logger_fn = logger_fn

        if weight_dir is not None:
            self.saver = tinder.saver.Saver(weight_dir, name)

            if load == "latest":
                self.saver.load_latest(self.bundle)
            elif load == "best":
                self.saver.load_best(self.bundle)
            elif isinstance(load, int):
                if not self.saver.load(self.bundle, epoch=load):
                    raise RuntimeError("fail to load epoch=", load)
            else:
                raise RuntimeError("cannot understand load arg", load)
        else:
            self.saver = None

    def train_epoch(
        self,
        loader,
        train_minibatch_fn,
        logging="step",
        manual_opt=False,
        interactive=False,
    ):
        assert (self.opt is not None) or manual_opt
        self.net.train()

        if interactive:
            loader = tqdm.tqdm(loader, leave=False)
        step = self.bundle.step

        n = 0
        metrics = {}
        for batch in loader:
            step += 1
            batch_size = len(batch)
            n += batch_size
            result = train_minibatch_fn(batch)

            if not manual_opt:
                self.opt.zero_grad()
                result["loss"].backward()
                self.opt.step()

            for (k, v) in result.items():
                if torch.is_tensor(v):
                    result[k] = v.item()

            if (self.logger_fn is not None) and logging == "step":
                for (k, v) in result.items():
                    # self.tb.add_scalar(k, v, step)
                    self.logger_fn("train." + k, v, step)

            if interactive:
                msg = "[train]"
                for (k, v) in result.items():
                    msg += f"    {k} = {v:8.4f}"
                loader.set_description(msg)

            for (k, v) in result.items():
                v *= batch_size
                if k in metrics:
                    metrics[k] += v
                else:
                    metrics[k] = v

        self.bundle.step = step
        self.bundle.epoch += 1

        # {
        #   'loss': ~,
        #   'acc': ~,
        # }
        for k in metrics:
            metrics[k] /= n

        if (self.logger_fn is not None) and logging == "epoch":
            for (k, v) in metrics.items():
                self.logger_fn("train." + k, v, step)

        # return per-sample metric
        return metrics

    def eval(self, loader, eval_minibatch_fn):
        self.net.eval()

        n = 0
        metrics = {}
        for batch in loader:
            n += len(batch)
            result = eval_minibatch_fn(batch)
            for (k, v) in result.items():
                if torch.is_tensor(v):
                    v = v.item()
                v *= len(batch)
                if k in metrics:
                    metrics[k] += v
                else:
                    metrics[k] = v
        for k in metrics:
            metrics[k] /= n

        step = self.bundle.step
        if self.logger_fn is not None:
            for (k, v) in metrics.items():
                self.logger_fn("eval." + k, v, step)
        return metrics

    def train(
        self,
        epochs,
        train_loader,
        train_minibatch_fn,
        logging="step",
        save_epoch_interval=1,
        score_col=None,
        manual_opt=False,
        eval_loader=None,
        eval_minibatch_fn=None,
        interactive=False,
    ):

        if interactive:
            YELLOW = Fore.YELLOW
            GREEN = Fore.GREEN
            RESET_ALL = Style.RESET_ALL
        else:
            YELLOW = ""
            GREEN = ""
            RESET_ALL = ""

        # for epoch in :
        for1 = (
            tqdm.trange(self.bundle.epoch + 1, epochs + 1, desc="epoch")
            if interactive
            else range(self.bundle.epoch + 1, epochs + 1)
        )
        for epoch in for1:
            metrics = self.train_epoch(
                train_loader,
                train_minibatch_fn,
                logging,
                manual_opt,
                interactive=interactive,
            )

            train_msg = ""
            for (k, v) in metrics.items():
                train_msg += f"{k} = {v:8.4f}    "

            if (eval_loader is not None) and (eval_minibatch_fn is not None):
                metrics = self.eval(eval_loader, eval_minibatch_fn)
                eval_msg = ""
                for (k, v) in metrics.items():
                    eval_msg += f"{k} = {v:8.4f}    "

                tqdm.tqdm.write(
                    f"epoch {epoch:3d} [train] {YELLOW}{train_msg}{RESET_ALL}"
                    + f"    [valid] {GREEN}{eval_msg}{RESET_ALL}"
                )
            else:
                tqdm.tqdm.write(
                    f"epoch {epoch:3d} [train] {YELLOW}{train_msg}{RESET_ALL}"
                )

            if (self.saver is not None) and (epoch % save_epoch_interval) == 0:
                self.saver.save(
                    self.bundle,
                    epoch=self.bundle.epoch,
                    score=metrics[score_col] if (score_col is not None) else 0.0,
                )

    def test(self, net, loader, test_minibatch_fn, id_col="id"):
        net.eval()
        result = {}

        for batch in loader:
            assert isinstance(batch, dict)
            assert id_col in batch

            batch["batch_size"] = len(batch)
            minibatch_result = test_minibatch_fn(batch)
            assert len(minibatch_result) == len(batch)
            for res in zip(batch[id_col], minibatch_result):
                result[res[0]] = res[1]

        return result
