import os
import torch

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
    def save(self, module, opt, epoch, score=None):
        """Save the model.

        `score` is used to choose the best model.
        An example for score is validation accuracy.

        Args:
            module (nn.Module): pytorch module
            opt (Optimizer, optional): optimizer. Defaults to None.
            epoch (int): number of epochs completed
            score (float, optional): Defaults to None. [description]
        """


        if score != None:
            if (self.best_score is None) or self.best_score < score:
                self.best_epoch = epoch
                self.best_score = score
                with open(self.dir_path + '/best_epoch') as f:
                    print(epoch, file=f)
                    print(score, file=f)

        dic = {
            'net': module.state_dict(),
            'epoch': epoch,
        }
        if opt:
            dic['opt'] = opt.state_dict()

        torch.save(dic, self.path_for_epoch(epoch))

    def load(self, module, opt, epoch):
        """Load the model.

        It is recommended to use `load_latest` or `load_best` instead.

        Args:
            module (nn.Module): model to load
            opt (Optimizer): optimizer to load
            epoch (int): epoch to load
        """

        dic = torch.load(
            self.path_for_epoch(epoch),
            map_location=lambda storage, loc: storage)

        module.load_state_dict(dic['net'])
        assert epoch == dic['epoch']

        if opt is not None:
            opt.load_state_dict(dic['opt'])

    def load_latest(self, module, opt) -> int:
        """Load the latest model.

        Args:
            module (nn.Module): model to load
            opt (Optimizer): optimizer to load

        Return:
            int: the epoch of the loaded model
        """

        latest:str = max(filter(lambda x: x.endswith('.pth'), os.listdir(self.dir_path)))
        assert latest.startswith('epoch_') and latest.endswith('.pth')
        epoch = int(latest[6:10])
        self.load(epoch, module, opt)
        return epoch


    def load_best(self, module, opt):
        """Load the best model.

        Args:
            module (nn.Module): model to load
            opt (Optimizer): optimizer to load
        """

        self.load(self.best_epoch, module, opt)
