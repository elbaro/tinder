import tinder
import numpy as np


def test_f1_score():
    c = tinder.metrics.ConfusionMatrix(3)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1, 1, 1, 0, 2, 2, 2])
    c.update(y_true, y_pred)

    assert c.f1_score_per_class() == np.array([0.5, 0.25, 0.25])
    assert c.f1_score_mean() == np.array([0.5, 0.25, 0.25]).mean()
