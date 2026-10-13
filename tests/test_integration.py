"""Integration tests with the real pretrained BLIP model and full pipeline."""

import os
from PIL import Image
import pytest

from src.blur_detection import assess_image_blur
from src.captioning import CaptionGenerator
from src.image_validation import validate_and_normalize_image
from src.restoration import ImageRestorationPipeline


@pytest.mark.integration
def test_real_model_loading_and_inference():
    """Integration test: Loads real BLIP model weights and performs live CPU inference."""
    generator = CaptionGenerator.get_instance()
    load_time = generator.load_model()
    assert generator.is_loaded is True
    assert generator.model is not None
    assert generator.processor is not None

    # Test image
    img_path = "data/images/scene_02_car_clean.png"
    if not os.path.exists(img_path):
        img = Image.new("RGB", (200, 200), color=(200, 30, 30))
    else:
        img = Image.open(img_path)

    res = generator.generate_caption(img, mode="concise")
    assert res.error is None
    assert len(res.caption) > 0
    assert res.inference_time_s > 0.0
    print(f"Integration caption: {res.caption}")


@pytest.mark.integration
def test_end_to_end_pipeline_flow():
    """
    Integration test: Runs complete flow:
    Upload bytes -> Validation -> Blur Check -> Restoration -> Real Captioning.
    """
    img_path = "data/images/scene_01_park_clean.png"
    assert os.path.exists(img_path), f"Sample image {img_path} missing."
    with open(img_path, "rb") as f:
        file_bytes = f.read()

    # 1. Validation
    val_res = validate_and_normalize_image(file_bytes)
    assert val_res.is_valid is True
    safe_img = val_res.safe_image

    # 2. Blur assessment
    assess_res = assess_image_blur(safe_img)
    assert 0.0 <= assess_res.clarity_score <= 100.0

    # 3. Restoration
    pipeline = ImageRestorationPipeline()
    rest_res = pipeline.restore_image(safe_img, force=True, method="adaptive")
    assert rest_res.enhanced_image is not None

    # 4. Captioning
    caption_gen = CaptionGenerator.get_instance()
    cap_res = caption_gen.generate_caption(rest_res.enhanced_image, mode="concise")
    assert cap_res.error is None
    assert len(cap_res.caption) > 0
