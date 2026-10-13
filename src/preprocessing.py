"""Preprocessing and image format conversion utilities."""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import get_logger

logger = get_logger("Preprocessing")


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Converts a PIL RGB image to a NumPy RGB uint8 array."""
    return np.array(image, dtype=np.uint8)


def cv2_to_pil(image_array: np.ndarray) -> Image.Image:
    """Converts a NumPy RGB uint8 array back to a PIL RGB Image."""
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array)


def safe_resize_for_inference(
    image: Image.Image,
    max_dimension: int = 1024
) -> Tuple[Image.Image, bool]:
    """
    Resizes an image preserving aspect ratio if either dimension exceeds max_dimension.
    Returns (resized_image, was_resized).
    """
    width, height = image.size
    if width <= max_dimension and height <= max_dimension:
        return image.copy(), False

    scale = max_dimension / max(width, height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    
    resized = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    logger.info(f"Scaled image from ({width}x{height}) to ({new_w}x{new_h}) for safe inference.")
    return resized, True


def create_motion_blur_kernel(kernel_size: int = 15, angle: float = 0.0) -> np.ndarray:
    """Generates a linear motion blur convolution kernel for evaluation."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = 1.0
    
    # Rotate kernel by specified angle
    rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def apply_synthetic_blur(
    image: Union[Image.Image, np.ndarray],
    blur_type: str = "motion",
    intensity: int = 11
) -> Image.Image:
    """
    Applies synthetic blur to an image for testing and evaluation purposes.
    Supported blur_type: 'gaussian' (defocus), 'motion', 'box'.
    """
    if isinstance(image, Image.Image):
        arr = pil_to_cv2(image)
    else:
        arr = image.copy()

    k_size = max(3, intensity if intensity % 2 == 1 else intensity + 1)

    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(arr, (k_size, k_size), sigmaX=k_size / 3.0)
    elif blur_type == "motion":
        kernel = create_motion_blur_kernel(k_size, angle=30.0)
        blurred = cv2.filter2D(arr, -1, kernel)
    elif blur_type == "box":
        blurred = cv2.blur(arr, (k_size, k_size))
    else:
        blurred = cv2.GaussianBlur(arr, (k_size, k_size), 0)

    return cv2_to_pil(blurred)
