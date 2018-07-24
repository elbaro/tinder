import tinder
import torch
import numpy as np


def test_f1_score_numpy():
    c = tinder.metrics.ConfusionMatrix(3)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1, 1, 1, 0, 2, 2, 2])
    c.update(y_true, y_pred)

    assert np.allclose(c.f1_score_per_class(), np.array([0.5, 0.25, 0.25]), rtol=1e-4)
    assert np.allclose(c.f1_score_mean(), np.array([0.5, 0.25, 0.25]).mean(), rtol=1e-4)


def test_f1_score_torch():
    c = tinder.metrics.ConfusionMatrix(3)
    y_true = torch.Tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = torch.Tensor([0, 2, 1, 0, 0, 1, 1, 1, 0, 2, 2, 2])
    c.update(y_true, y_pred)

    assert np.allclose(c.f1_score_per_class(), np.array([0.5, 0.25, 0.25]), rtol=1e-4)
    assert np.allclose(c.f1_score_mean(), np.array([0.5, 0.25, 0.25]).mean(), rtol=1e-4)
