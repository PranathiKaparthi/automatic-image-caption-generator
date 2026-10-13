"""Unit tests for blur detection and optical assessment algorithms."""

import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.blur_detection import (
    assess_image_blur,
    compute_fft_high_frequency_ratio,
    compute_laplacian_variance,
    compute_tenengrad_score,
)
from src.preprocessing import apply_synthetic_blur


@pytest.fixture
def sharp_checkerboard_image() -> Image.Image:
    """Creates a high-contrast sharp checkerboard image."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(0, 200, 20):
        for y in range(0, 200, 20):
            if (x // 20 + y // 20) % 2 == 0:
                draw.rectangle([x, y, x + 20, y + 20], fill=(0, 0, 0))
    return img


def test_sharp_image_assessment(sharp_checkerboard_image):
    """Verifies that high-contrast sharp images are recognized as sharp."""
    result = assess_image_blur(sharp_checkerboard_image)
    assert result.is_blurry is False
    assert result.blur_level == "sharp"
    assert result.clarity_score > 60.0
    assert result.laplacian_variance > 100.0


def test_motion_blurred_image(sharp_checkerboard_image):
    """Verifies that synthetic motion blur drops clarity and is flagged as blurry on natural scenes."""
    # Test on natural scene pattern
    nat_img = Image.new("RGB", (200, 200), color=(140, 180, 220))
    draw = ImageDraw.Draw(nat_img)
    draw.rectangle([20, 100, 180, 190], fill=(40, 120, 40))
    draw.ellipse([50, 40, 150, 120], fill=(220, 60, 60))

    clean_res = assess_image_blur(nat_img)
    blurred = apply_synthetic_blur(nat_img, blur_type="motion", intensity=15)
    blurred_res = assess_image_blur(blurred)

    assert blurred_res.is_blurry is True
    assert blurred_res.blur_level in ("mild_blur", "severe_blur")
    assert blurred_res.laplacian_variance < clean_res.laplacian_variance


def test_defocus_blurred_image(sharp_checkerboard_image):
    """Verifies that defocus / Gaussian blur reduces Laplacian and FFT high-frequencies."""
    blurred = apply_synthetic_blur(sharp_checkerboard_image, blur_type="gaussian", intensity=15)
    result = assess_image_blur(blurred)
    assert result.is_blurry is True
    assert result.laplacian_variance < 100.0


def test_clarity_score_bounds(sharp_checkerboard_image):
    """Verifies clarity score always strictly stays within [0.0, 100.0]."""
    res1 = assess_image_blur(sharp_checkerboard_image)
    assert 0.0 <= res1.clarity_score <= 100.0

    flat_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    res2 = assess_image_blur(flat_img)
    assert 0.0 <= res2.clarity_score <= 100.0
    assert res2.is_blurry is True
