import numpy as np
from typing import NamedTuple, List
from PIL import Image


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
    def from_another(another: 'BoundingBox'):
        return BoundingBox.from_points(
            another.left, another.top, another.right, another.bottom, another.max_width,
            another.max_height)

    @staticmethod
    def from_points(left: float, top: float, right: float, bottom: float, max_width: int,
                    max_height: int) -> 'BoundingBox':
        return BoundingBox(left, top, right, bottom, right - left, bottom - top, max_width, max_height)

    @staticmethod
    def from_size(left: float, top: float, width: float, height: float, max_width: int, max_height: int) -> 'BoundingBox':
        return BoundingBox(left, top, left + width, top + height, width, height, max_width, max_height)

    def stretch_by_ratio(self, left: float, top: float, right: float, bottom: float):
        left = max(0, self.left - self.width * left)
        right = min(self.max_width, self.right + self.width * right)

        top = max(0, self.top - self.height * top)
        bottom = min(self.max_height, self.bottom + self.height * bottom)

        return BoundingBox.from_points(left, top, right, bottom, self.max_width, self.max_height)

    def stretch_by(self, left: float, top: float, right: float, bottom: float):
        left = max(0, self.left - left)
        right = min(self.max_width, self.right + right)

        top = max(0, self.top - top)
        bottom = min(self.max_height, self.bottom + bottom)

        return BoundingBox.from_points(left, top, right, bottom, self.max_width, self.max_height)

    def int(self):
        left = int(self.left)
        top = int(self.top)
        right = int(self.right)
        bottom = int(self.bottom)
        return BoundingBox.from_points(left, top, right, bottom, self.max_width, self.max_height)


def crop(img: np.ndarray, crop: BoundingBox, boxes_to_transform: List['BoundingBox'] = None):
    crop = crop.int()
    img = img[crop.top:crop.bottom, crop.left:crop.right]

    if boxes_to_transform:
        return img, \
            [BoundingBox.from_size(
                box.left - crop.left,
                box.top - crop.top,
                box.width,
                box.height,
                crop.width,  # already int
                crop.height)  # already int
             for box in boxes_to_transform]

    return img


def filename_to_normalized_rgb(filename: str, wh):
    img = Image.open(filename).convert('RGB').resize(wh)
    img = np.asarray(img)
    img = np.transpose(img, axes=(2, 0, 1)).astype(np.float32) / 255.0 * 2 - 1  # [C,H,W], -1~1
    return img
