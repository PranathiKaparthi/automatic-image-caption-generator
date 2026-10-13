"""Unit tests for optical and deep restoration pipelines."""

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from src.blur_detection import assess_image_blur
from src.preprocessing import apply_synthetic_blur, pil_to_cv2
from src.restoration import (
    DeblurNetLite,
    ImageRestorationPipeline,
    bilateral_unsharp_restoration,
    richardson_lucy_deconvolution,
    wiener_deconvolution,
)


@pytest.fixture
def synthetic_scene():
    """Generates a small test pattern."""
    img = Image.new("RGB", (120, 120), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 90, 90], fill=(20, 20, 120))
    draw.ellipse([40, 40, 80, 80], fill=(220, 50, 50))
    return img


def test_richardson_lucy_deconvolution(synthetic_scene):
    """Verifies that Richardson-Lucy deconvolution returns array with identical shape and type."""
    arr = pil_to_cv2(synthetic_scene)
    psf = np.ones((5, 5), dtype=np.float32) / 25.0
    deblurred = richardson_lucy_deconvolution(arr, psf, iterations=5)
    assert deblurred.shape == arr.shape
    assert deblurred.dtype == np.uint8


def test_wiener_deconvolution(synthetic_scene):
    """Verifies Wiener frequency deconvolution execution."""
    arr = pil_to_cv2(synthetic_scene)
    psf = np.ones((5, 5), dtype=np.float32) / 25.0
    deblurred = wiener_deconvolution(arr, psf, noise_ratio=0.01)
    assert deblurred.shape == arr.shape
    assert deblurred.dtype == np.uint8


def test_bilateral_unsharp_restoration(synthetic_scene):
    """Verifies edge-preserving bilateral unsharp enhancement."""
    arr = pil_to_cv2(synthetic_scene)
    enhanced = bilateral_unsharp_restoration(arr)
    assert enhanced.shape == arr.shape
    assert enhanced.dtype == np.uint8


def test_deblurnet_lite_forward():
    """Verifies PyTorch DeblurNetLite model tensor shape and grad-free execution."""
    model = DeblurNetLite(width=16, num_blocks=2)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


def test_restoration_pipeline_preserves_original(synthetic_scene):
    """Verifies original input image object remains unaltered."""
    pipeline = ImageRestorationPipeline()
    orig_copy = synthetic_scene.copy()
    
    blurred = apply_synthetic_blur(synthetic_scene, blur_type="gaussian", intensity=9)
    res = pipeline.restore_image(blurred, force=True, method="guided")
    
    # Original image must be preserved
    assert res.enhanced_image is not None
    assert synthetic_scene.tobytes() == orig_copy.tobytes()
    assert res.enhanced_image.size == blurred.size


def test_restoration_skips_sharp_images(synthetic_scene):
    """Verifies restoration is skipped for already sharp images unless forced."""
    pipeline = ImageRestorationPipeline()
    res = pipeline.restore_image(synthetic_scene, force=False)
    assert res.is_restored is False
    assert "already sharp" in res.method_used.lower()
