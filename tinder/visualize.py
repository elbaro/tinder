import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pylab

cmap = plt.get_cmap('Accent', 256)

def show_imgs(imgs, rows, cols, is_indexed:[bool]=None):
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
        is_indexed ([bool]): a list of flags that represents whether images are indexed images.
    """

    assert len(imgs) <= rows*cols
    if is_indexed is None: is_indexed = [False] * len(imgs)
    assert len(imgs) == len(is_indexed)

    fig = plt.figure()

    for i, (img, cmap) in enumerate(zip(imgs, is_indexed)):
        if type(img) == torch.Tensor:
            img = img.detach().cpu().numpy()

        if img.shape[0]==3:
            img = np.transpose(img, axes=(1,2,0))

        fig.add_subplot(rows,cols,i+1)

        if is_indexed:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)

    plt.show()


# def colorize(imgs, colormap=None):
#     """Colorize indexed images.

#     Useful for visualizing segmentation labels.

#     Args:
#         imgs ([type]): indexed images.
#         colormap ([type], optional): Defaults to None. [description]
#     """

#     vals = np.linspace(0,1,256)
#     np.random.shuffle(vals)
#     cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))