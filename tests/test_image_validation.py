"""Unit tests for image validation and normalization."""

import io
from PIL import Image
import pytest

from src.config import ImageConfig
from src.image_validation import validate_and_normalize_image


def create_test_image_bytes(format_name="PNG", size=(200, 200), mode="RGB", color=(100, 150, 200)):
    """Helper to generate in-memory image bytes."""
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def test_valid_jpeg_and_png():
    """Verifies that standard JPEG and PNG files are accepted and normalized."""
    png_bytes = create_test_image_bytes("PNG")
    res_png = validate_and_normalize_image(png_bytes)
    assert res_png.is_valid is True
    assert res_png.safe_image is not None
    assert res_png.safe_image.mode == "RGB"
    assert res_png.original_format == "PNG"

    jpg_bytes = create_test_image_bytes("JPEG")
    res_jpg = validate_and_normalize_image(jpg_bytes)
    assert res_jpg.is_valid is True
    assert res_jpg.safe_image.mode == "RGB"


def test_valid_webp():
    """Verifies that WebP images are supported."""
    webp_bytes = create_test_image_bytes("WEBP")
    res = validate_and_normalize_image(webp_bytes)
    assert res.is_valid is True
    assert res.original_format == "WEBP"


def test_rgba_normalization_and_white_blend():
    """Verifies RGBA images are converted to standard RGB without alpha errors."""
    rgba_bytes = create_test_image_bytes("PNG", mode="RGBA", color=(255, 0, 0, 128))
    res = validate_and_normalize_image(rgba_bytes)
    assert res.is_valid is True
    assert res.safe_image.mode == "RGB"
    assert res.warning_message is not None
    assert "RGBA" in res.warning_message


def test_empty_upload():
    """Verifies that zero-byte files are rejected with informative error."""
    res = validate_and_normalize_image(b"")
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()


def test_corrupted_file():
    """Verifies that corrupted random binary payloads are safely caught."""
    corrupt_bytes = b"NOT_AN_IMAGE_HEADER_1234567890abcdef"
    res = validate_and_normalize_image(corrupt_bytes)
    assert res.is_valid is False
    assert res.error_message is not None


def test_unsupported_format_gif():
    """Verifies that formats not in allowed_formats (like GIF) are rejected."""
    gif_bytes = create_test_image_bytes("GIF")
    res = validate_and_normalize_image(gif_bytes)
    assert res.is_valid is False
    assert "unsupported" in res.error_message.lower()


def test_file_size_limit():
    """Verifies that files exceeding maximum allowed bytes are rejected."""
    cfg = ImageConfig(max_file_size_bytes=500)
    img_bytes = create_test_image_bytes("JPEG", size=(500, 500))
    res = validate_and_normalize_image(img_bytes, config=cfg)
    assert res.is_valid is False
    assert "file size exceeds" in res.error_message.lower()


def test_dimension_limit():
    """Verifies that images exceeding maximum dimensions are rejected."""
    cfg = ImageConfig(max_dimensions=(100, 100))
    img_bytes = create_test_image_bytes("PNG", size=(150, 80))
    res = validate_and_normalize_image(img_bytes, config=cfg)
    assert res.is_valid is False
    assert "dimensions" in res.error_message.lower()


def test_original_bytes_preservation():
    """Verifies that input byte string is completely immutable."""
    original_bytes = create_test_image_bytes("PNG")
    bytes_copy = bytes(original_bytes)
    res = validate_and_normalize_image(original_bytes)
    assert res.is_valid is True
    assert original_bytes == bytes_copy
