import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pylab

default_cmap = plt.get_cmap('Accent', 256)


def show_imgs(imgs, rows=None, cols=None, cmaps: []=None):
    """
    Show an image grid in a popup window.

    imgs can be a list of

    - torch.Tensor or numpy.ndarray
    - 0~1 or 0~255
    - HWC or CWH
    - None (skip the grid cell for alignmentt)

    Args:
        imgs
        rows
        cols
        cmaps ([objects]): a list of matplot ListedColormaps or booleans. If True, use the default cmap. If False, colormap is not used. If ListedColormap is provided, use it.
    """

    if not isinstance(imgs, list):
        imgs = [imgs]

    n = len(imgs)
    if rows is None and cols is None:
        rows = int(n**0.5)
        cols = (n-1)//rows+1
    elif rows is None:
        cols = (n-1)//rows+1
    elif cols is None:
        rows = (n-1)//cols+1

    assert len(imgs) <= rows*cols

    if cmaps is None:
        cmaps = [False] * len(imgs)
    assert len(imgs) == len(cmaps)

    fig = plt.figure()

    for i, (img, cmap) in enumerate(zip(imgs, cmaps)):
        if img is None:
            continue

        if type(img) == torch.Tensor:
            img = img.detach().cpu().numpy()

        if img.shape[0] == 3:
            img = np.transpose(img, axes=(1, 2, 0))

        fig.add_subplot(rows, cols, i+1)

        if cmap is True:
            plt.imshow(img, cmap=default_cmap)
        elif cmap is False or cmap is None:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=cmap, vmin=0, vmax=len(cmap.colors))

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
