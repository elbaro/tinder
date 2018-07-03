import os
import torch
import urllib
import time
import sys


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def assert_download(weight_url, weight_dest):
    if not os.path.exists(weight_dest):
        if weight_url:
            print('downloading weight:')
            print('    ' + weight_url)
            print('    ' + weight_dest)
            urllib.request.urlretrieve(weight_url, weight_dest, reporthook=_reporthook)
        else:
            raise NotImplementedError("please specify url to download in your model")


class Saver(object):
    """A helper class to save and load your model.

    Example::

        saver = Saver('/data/weights/', 'resnet152-cosine')
        saver.load_latest(alexnet, opt)  # resume from the latest
        for epoch in range(100):
            ..
            saver.save(alexnet, opt, epoch=epoch, score=acc)

        # inference
        saver.load_best(alexnet, opt=None) # no need for optimizer

    The batch dimension is implicit.
    The above code is the same as `tensor.view(tensor.size(0), 3, -1, 256)`.

    Args:
        weight_dir (str): directory for your weights
        exp_name (str): name of your experiment (e.g. resnet152-cosine)
    """

    def __init__(self, weight_dir, exp_name):
        self.weight_dir = weight_dir
        self.exp_name = exp_name
        self.dir_path = weight_dir + '/' + exp_name
        os.makedirs(self.dir_path, exist_ok=True)

        self.best_epoch_path = self.dir_path + '/best_epoch'
        if os.path.exists(self.best_epoch_path):
            with open(self.best_epoch_path) as f:
                self.best_epoch = int(f.readline())
                self.best_score = float(f.readline())
        else:
            self.best_epoch = None
            self.best_score = None

    def path_for_epoch(self, epoch):
        return self.dir_path + '/' + 'epoch_%04d.pth' % epoch

    # ex. ~/imagenet/weights/alexnet/epoch_0001.pth
    def save(self, dic: dict, epoch: int, score: float=None):
        """Save the model.

        `score` is used to choose the best model.
        An example for score is validation accuracy.

        Example::

            saver = Saver()
            saver.save(
                {
                    'net':net,
                    'opt':opt,
                    'scheduler':cosine_annealing
                },
                epoch=3,
                score=val_acc
            )

        Args:
            dic (dict): the values are objects with `state_dict` and `load_state_dict`
            epoch (int): number of epochs completed
            score (float, optional): Defaults to None
        """

        if score != None:
            if (self.best_score is None) or self.best_score < score:
                self.best_epoch = epoch
                self.best_score = score
                with open(self.dir_path + '/best_epoch', 'w') as f:
                    print(epoch, file=f)
                    print(score, file=f)

        new_dic = {}
        for key, value in dic.items():
            new_dic[key] = value.state_dict()
        new_dic['epoch'] = epoch

        torch.save(new_dic, self.path_for_epoch(epoch))

    def load(self, dic: dict, epoch: int):
        """Load the model.

        It is recommended to use `load_latest` or `load_best` instead.

        Args:
            dic (dict): see save()
            epoch (int): epoch to load
        """

        states = torch.load(
            self.path_for_epoch(epoch),
            map_location=lambda storage, loc: storage)

        assert epoch == states['epoch']

        for key, value in dic.items():
            value.load_state_dict(states[key])

    def load_latest(self, dic: dict) -> int:
        """Load the latest model.

        Args:
            dic (dict): see save()

        Return:
            int: the epoch of the loaded model. 0 if no model exists.
        """

        files = list(filter(lambda x: x.endswith('.pth'), os.listdir(self.dir_path)))
        if len(files) == 0:
            return 0

        latest: str = max(files)
        assert latest.startswith('epoch_') and latest.endswith('.pth')
        epoch = int(latest[6:10])
        self.load(dic, epoch)

        return epoch

    def load_best(self, dic: dict) -> bool:
        """Load the best model.

        Args:
            dic (dict): see save()

        Return:
            bool: True if loaded successfully
        """

        if self.best_epoch is not None:
            self.load(dic, self.best_epoch)
            return True

        return False
