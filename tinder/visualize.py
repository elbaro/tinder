import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


def show_imgs(imgs, rows, cols):
    """
    Show an image grid in a popup window.

    imgs can be a list of

    - torch.Tensor or numpy.ndarray
    - 0~1 or 0~255
    - HWC or CWH

    Args:
        imgs
        rows
        cols
    """

    assert len(imgs) <= rows*cols

    fig = plt.figure()

    for i, img in enumerate(imgs):
        if type(img) == torch.Tensor:
            img = img.detach().cpu().numpy()

        if img.shape[0]==3:
            img = np.transpose(img, axes=(1,2,0))

        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)

    plt.show()
