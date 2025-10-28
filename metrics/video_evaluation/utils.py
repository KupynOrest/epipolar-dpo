import cv2
import numpy as np


def resize_and_center_crop(image: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Resize the image so that the smallest side is 'size' pixels,
    then perform a center crop to make it square.

    Args:
        image: Input image (numpy array)
        size: Target size for both width and height

    Returns:
        Center-cropped image of size (size, size)
    """
    h, w = image.shape[:2]
    scale = size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    start_y = max(0, new_h // 2 - size // 2)
    start_x = max(0, new_w // 2 - size // 2)
    cropped = resized[start_y:start_y + size, start_x:start_x + size]

    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size))

    return cropped
