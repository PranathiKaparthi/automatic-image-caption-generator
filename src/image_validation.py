"""Image validation and safe normalization module."""

from dataclasses import dataclass
import io
from typing import Optional, Tuple
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import ImageConfig, get_logger

logger = get_logger("ImageValidation")


@dataclass
class ValidationResult:
    """Outcome of validating an uploaded image."""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    original_format: Optional[str] = None
    original_size: Optional[Tuple[int, int]] = None  # (width, height)
    file_size_bytes: int = 0
    safe_image: Optional[Image.Image] = None  # Normalized RGB PIL image for inference


def validate_and_normalize_image(
    file_bytes: bytes,
    config: Optional[ImageConfig] = None
) -> ValidationResult:
    """
    Safely validates uploaded image bytes against size, format, decompression-bomb risks,
    and returns a normalized RGB PIL image. The original bytes are never mutated.

    Args:
        file_bytes: Raw binary data of the uploaded file.
        config: Validation parameters (defaults to ImageConfig()).

    Returns:
        ValidationResult object indicating success/failure and normalized image.
    """
    if config is None:
        config = ImageConfig()

    file_size = len(file_bytes) if file_bytes else 0
    if file_size == 0:
        return ValidationResult(
            is_valid=False,
            error_message="Uploaded file is empty (0 bytes). Please upload a valid image.",
            file_size_bytes=0,
        )

    if file_size > config.max_file_size_bytes:
        max_mb = config.max_file_size_bytes / (1024 * 1024)
        current_mb = file_size / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error_message=f"File size exceeds limit: {current_mb:.2f} MB > {max_mb:.1f} MB allowed.",
            file_size_bytes=file_size,
        )

    # Enforce decompression bomb threshold
    Image.MAX_IMAGE_PIXELS = config.safe_pixel_limit

    try:
        # Open in memory
        stream = io.BytesIO(file_bytes)
        img = Image.open(stream)
        
        # Verify image structure by reading metadata / format
        img_format = img.format.upper() if img.format else "UNKNOWN"
        width, height = img.size

        if img_format not in [fmt.upper() for fmt in config.allowed_formats]:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Unsupported image format '{img_format}'. "
                    f"Supported formats: {', '.join(config.allowed_formats)}."
                ),
                file_size_bytes=file_size,
                original_format=img_format,
                original_size=(width, height),
            )

        # Verify pixel dimensions
        max_w, max_h = config.max_dimensions
        if width > max_w or height > max_h:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image dimensions ({width}x{height}) exceed maximum allowed ({max_w}x{max_h}).",
                file_size_bytes=file_size,
                original_format=img_format,
                original_size=(width, height),
            )

        # Test raster load to catch truncated or corrupted payload
        img.load()

        # Handle EXIF orientation if present
        try:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
        except Exception as e:
            logger.debug(f"Exif transpose skipped: {e}")

        warning = None
        # Safe normalization to RGB (e.g. RGBA, CMYK, Palette, Grayscale)
        if img.mode == "RGBA":
            # Composite onto white background to avoid black background artifacts from transparent areas
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            normalized_img = background
            warning = "Converted RGBA image to RGB with white background blending."
        elif img.mode != "RGB":
            normalized_img = img.convert("RGB")
            if img.mode != "L":
                warning = f"Converted image mode '{img.mode}' to standard RGB."
        else:
            # Create an independent copy to protect against mutations
            normalized_img = img.copy()

        logger.info(f"Image validated successfully: format={img_format}, size={width}x{height}, bytes={file_size}")
        return ValidationResult(
            is_valid=True,
            original_format=img_format,
            original_size=(width, height),
            file_size_bytes=file_size,
            safe_image=normalized_img,
            warning_message=warning,
        )

    except Image.DecompressionBombError:
        return ValidationResult(
            is_valid=False,
            error_message="Image exceeds safe pixel density (decompression bomb protection triggered).",
            file_size_bytes=file_size,
        )
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Corrupted or invalid image file: {str(e)}",
            file_size_bytes=file_size,
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Unexpected error while validating image: {str(e)}",
            file_size_bytes=file_size,
        )
