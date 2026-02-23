import cv2
import grain.python as grain
import numpy as np


class CenterCropAndResize(grain.MapTransform):
    """Center-crops the image to a square and resizes to the target size.

    Args:
        target_size: The target size of the image.
    """

    def __init__(self, target_size: int):
        self.target_size = target_size

    def map(self, element: dict) -> dict:
        image = element["image"]  # (H, W, C) uint8 numpy array
        h, w = image.shape[:2]
        min_side = min(h, w)
        # Center crop to square
        top = (h - min_side) // 2
        left = (w - min_side) // 2
        image = image[top : top + min_side, left : left + min_side]
        # Resize with area interpolation (similar to tf antialias)
        image = cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        element["image"] = image
        return element


class RandomFlipLeftRight(grain.RandomMapTransform):
    """Randomly flips the image horizontally."""

    def random_map(self, element: dict, rng: np.random.Generator) -> dict:
        if rng.integers(2):
            element["image"] = np.ascontiguousarray(element["image"][:, ::-1])
        return element


class NormalizeImage(grain.MapTransform):
    """Casts to float32 and normalizes pixel values to [-1, 1]."""

    def map(self, element: dict) -> dict:
        image = element["image"].astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        element["image"] = image
        return element


class ExtractImageAndLabel(grain.MapTransform):
    """Extracts (image, label) tuple from the element dict."""

    def map(self, element: dict) -> tuple[np.ndarray, np.ndarray]:
        return element["image"], element["label"]
