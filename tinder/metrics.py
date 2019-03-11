import torch
import numpy as np


def instance_segmentation_iou(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError()
    pass


def semantic_segmentation_iou(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """

    Calculate IOU per class.

    IOU (Intersection Over Union) = true positive / (TP + FP + FN)
    IOU = # (predict == label == class_i) / (predict==class_i or label==class_i)

    Args:
        prediction: [N, C, H, W] tensor
        label: [N, H, W] tensor

    Returns:
        (torch.Tensor): [N, C] tensor

    """

    n = score.shape[0]
    num_classes = score.shape[1]

    _, prediction = score.detach().max(dim=1)
    prediction = prediction.view(n, -1)
    label = label.detach().view(n, -1)

    out = []
    for i in range(num_classes):
        TP = ((prediction == i) & (label == i)).float().sum(dim=1)  # [N]
        union = ((prediction == i) | (label == i)).float().sum(dim=1) + 1  # [N]
        out.append(TP / union)

    out = torch.stack(out, dim=1)
    return out  # [N, C]


class Mean(object):
    pass


class Accuracy(object):
    pass


class ConfusionMatrix(object):
    EPS = 1e-5

    def __init__(self, num_cls):
        self.num_cls = num_cls
        self.TP = np.zeros(self.num_cls)
        self.TP_FN = np.zeros(self.num_cls)
        self.TP_FP = np.zeros(self.num_cls)

    def clear(self):
        self.TP.fill(0)
        self.TP_FN.fill(0)
        self.TP_FP.fill(0)

    def update(self, prediction, answer):
        """[summary]

        Args:
            prediction ([type]): A torch.Tensor or numpy.ndarray of [N,C,...]
            answer ([type]): A torch.Tensor or numpy.ndarray of [N,C,...]
        """

        # to # [N,C,?]
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.view(-1)
            answer = answer.view(-1)

            for c in range(self.num_cls):
                self.TP[c] += ((prediction == c) & (answer == c)).int().sum().item()
                self.TP_FP[c] += (prediction == c).int().sum().item()
                self.TP_FN[c] += (answer == c).int().sum().item()

        elif isinstance(prediction, np.ndarray):
            prediction = prediction.reshape(-1)
            answer = answer.reshape(-1)

            for c in range(self.num_cls):
                self.TP[c] += ((prediction == c) & (answer == c)).sum()
                self.TP_FP[c] += (prediction == c).sum()
                self.TP_FN[c] += (answer == c).sum()

    def precision(self):
        return self.TP / (self.TP_FP + self.EPS)

    def recall(self):
        return self.TP / (self.TP_FN + self.EPS)

    def f1_score_per_class(self):
        precision = self.precision()
        recall = self.recall()
        return 2 / ((1 / (precision + self.EPS)) + (1 / (recall + self.EPS)))

    def f1_score_mean(self):
        return self.f1_score_per_class().mean()
