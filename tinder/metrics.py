import torch

def instance_segmentation_iou(score: torch.Tensor, label: torch.Tensor):
    raise NotImplementedError()
    pass

def semantic_segmentation_iou(score: torch.Tensor, label: torch.Tensor):
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

    for i in range(num_classes):
        TP = (prediction == i and label == i).int().sum(dim=2)    # [N, C]
        union = (prediction == i or label == i).int().sum(dim=2)  # [N, C]

    return TP / union
