import numpy as np
from typing import NamedTuple, List


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
        right = min(self.max_width, self.left + self.width + self.width * right)

        top = max(0, self.top - self.height * top)
        bottom = min(self.max_height, self.top + self.height + self.height * bottom)

        return BoundingBox.from_points(left, top, right, bottom, self.max_width, self.max_height)

    def int(self):
        self.left = int(self.left)
        self.top = int(self.top)
        self.right = int(self.right)
        self.bottom = int(self.bottom)
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        return self


def crop(img: np.ndarray, crop: BoundingBox, boxes_to_transform: List['BoundingBox'] = None):
    crop = BoundingBox.from_another(crop).int()
    img = img[crop.top:crop.bottom, crop.left:crop.right]

    if boxes_to_transform:
        return img, \
               [BoundingBox.from_size(
                   box.left - crop.left,
                   box.top - crop.top,
                   box.width,
                   box.height,
                   crop.width,
                   crop.height)
                   for box in boxes_to_transform]

    return img
