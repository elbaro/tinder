from torch import nn
from torchvision import transforms
import sacred
import tinder
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import sacred
from sacred.stflow import LogFileWriter
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

ex = sacred.Experiment("mnist")
# ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(MongoObserver.create())
# ex.observers.append(FileStorageObserver.create('my_runs'))

net = None


@ex.config
def config():
    name = "sample"
    weight_dir = "weights"
    epochs = 100
    lr = 0.001
    device = "cuda"
    num_workers = 16
    batch_size = 32


@ex.config_hook
def hook(config, command_name, logger):
    pass


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@ex.capture
def get_model(name, weight_dir, lr, seed, device):
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    model = tinder.model.Model(
        net=net, name=name, opt=opt, logger_fn=ex.log_scalar, weight_dir=weight_dir
    )
    return model


@ex.automain
# @LogFileWriter(ex)
def train(epochs, batch_size, num_workers, device, seed):
    model = get_model()
    dataset = torchvision.datasets.MNIST(
        root="mnist",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    d_train, d_valid = tinder.dataset.random_split(dataset, [0.8], seed=seed)
    loader_train = torch.utils.data.DataLoader(
        d_train, batch_size, shuffle=True, num_workers=num_workers
    )
    loader_valid = torch.utils.data.DataLoader(
        d_valid, batch_size, shuffle=True, num_workers=num_workers
    )

    def train_minibatch_fn(batch):
        img = batch[0].to(device)
        labels = batch[1].to(device)
        logits = model.net(img)
        loss = F.cross_entropy(input=logits, target=labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        return {"loss": loss, "acc": acc}

    print("training ..")
    model.train(
        epochs=epochs,
        train_loader=loader_train,
        train_minibatch_fn=train_minibatch_fn,
        eval_loader=loader_valid,
        eval_minibatch_fn=train_minibatch_fn,
        score_col="acc",
        interactive=False,
    )


@ex.command
# @LogFileWriter(ex)
def test(batch_size, num_workers):
    model = get_model()
    d_test = torchvision.datasets.MNIST(root="mnist", download=True)
    loader = torch.utils.data.DataLoader(
        d_test, batch_size, shuffle=True, num_workers=num_workers
    )

    def test_minibatch_fn(batch):
        img = batch[0].to(device)
        logits = model.net(img)
        return logits.argmax(dim=1)

    result = model.test(loader=loader, test_minibatch_fn=test_minibatch_fn)

    with open("result.txt", "w") as f:
        for (k, v) in result.items():
            f.write(f"{k} {v}\n")
