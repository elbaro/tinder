import torch
import torchvision
import numpy as np
from typing import NamedTuple, List
from PIL import Image

if "post" not in Image.PILLOW_VERSION:
    print("[tinder warning] You have /pillow/ instead of /pillow-simd/.")


class BoundingBox(NamedTuple):
    """
    A class to represent a rectangular region in a image.

    You are encouraged to use `from_` constructors.
    """

    left: float
    top: float
    right: float
    bottom: float

    width: float
    height: float

    max_width: int
    max_height: int

    @staticmethod
    def from_another(another: "BoundingBox"):
        return BoundingBox.from_points(
            another.left,
            another.top,
            another.right,
            another.bottom,
            another.max_width,
            another.max_height,
        )

    @staticmethod
    def from_points(
        left: float,
        top: float,
        right: float,
        bottom: float,
        max_width: int,
        max_height: int,
    ) -> "BoundingBox":
        return BoundingBox(
            left, top, right, bottom, right - left, bottom - top, max_width, max_height
        )

    @staticmethod
    def from_size(
        left: float,
        top: float,
        width: float,
        height: float,
        max_width: int,
        max_height: int,
    ) -> "BoundingBox":
        return BoundingBox(
            left, top, left + width, top + height, width, height, max_width, max_height
        )

    def stretch_by_ratio(self, left: float, top: float, right: float, bottom: float):
        left = max(0, self.left - self.width * left)
        right = min(self.max_width, self.right + self.width * right)

        top = max(0, self.top - self.height * top)
        bottom = min(self.max_height, self.bottom + self.height * bottom)

        return BoundingBox.from_points(
            left, top, right, bottom, self.max_width, self.max_height
        )

    def stretch_by(self, left: float, top: float, right: float, bottom: float):
        left = max(0, self.left - left)
        right = min(self.max_width, self.right + right)

        top = max(0, self.top - top)
        bottom = min(self.max_height, self.bottom + bottom)

        return BoundingBox.from_points(
            left, top, right, bottom, self.max_width, self.max_height
        )

    def int(self):
        left = int(self.left)
        top = int(self.top)
        right = int(self.right)
        bottom = int(self.bottom)
        return BoundingBox.from_points(
            left, top, right, bottom, self.max_width, self.max_height
        )


def crop(
    img: np.ndarray, crop: BoundingBox, boxes_to_transform: List["BoundingBox"] = None
):
    crop = crop.int()
    img = img[crop.top : crop.bottom, crop.left : crop.right]

    if boxes_to_transform:
        return (
            img,
            [
                BoundingBox.from_size(
                    box.left - crop.left,
                    box.top - crop.top,
                    box.width,
                    box.height,
                    crop.width,  # already int
                    crop.height,
                )  # already int
                for box in boxes_to_transform
            ],
        )

    return img


def fft2d_log(img: np.ndarray):
    from scipy import fftpack

    if len(img.shape) == 3:
        img = img.mean(axis=2)
    f = fftpack.fft2(img)
    f = fftpack.fftshift(f)
    return np.log(abs(f))


def fft2d(img: np.ndarray):
    from scipy import fftpack

    if len(img.shape) == 3:
        img = img.mean(axis=2)
    f = fftpack.fft2(img)
    f = fftpack.fftshift(f)
    return abs(f)


def pggan_bbox_from_landmarks(wh, e0e1m0m1s):
    """[summary]

    Args:
        wh (tuple): (width, height)
        e0e1m0m1s (list): [ [ e0,e1,m0,m1] ]

    Returns:
        A list of ([[x,y],[x,y],[x,y],[x,y]] in ccw, is_oversize)
    """

    ret = []
    for (e0, e1, m0, m1) in e0e1m0m1s:

        xx = e1 - e0
        yy = (e0 + e1) / 2 - (m0 + m1) / 2

        c = (e0 + e1) / 2 - 0.1 * yy
        s = max(4.0 * np.linalg.norm(xx), 3.6 * np.linalg.norm(yy))

        x = np.array((xx[0] - yy[1], xx[1] + yy[0]))
        x /= np.linalg.norm(x)
        y = np.array((-x[1], x[0]))

        # ccw
        crop = [
            (c - x * s / 2 - y * s / 2).tolist(),
            (c - x * s / 2 + y * s / 2).tolist(),
            (c + x * s / 2 + y * s / 2).tolist(),
            (c + x * s / 2 - y * s / 2).tolist(),
        ]

        oversize = (
            min(z for p in crop for z in p) < 0
            or max(p[0] for p in crop) >= wh[0]
            or max(p[1] for p in crop) >= wh[1]
        )
        ret.append((crop, oversize))

    return ret


def load_rgb(
    path,
    width,
    height,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    pil_transform=None,
    numpy_transform=None,
) -> torch.Tensor:
    img = Image.open(path).resize((width, height))
    if img.mode != "RGB":
        img = img.convert("RGB")

    if pil_transform is not None:
        img = pil_transform(img)

    img = np.asarray(img)  # [H,W,3]
    if numpy_transform is not None:
        img = numpy_transform(img)

    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3,H,W], 0~1
    tensor = torchvision.transforms.functional.normalize(
        tensor, mean, std
    )  # about -1~1
    return tensor
