"""Configuration management for Image Caption Generator with Deblurring."""

from dataclasses import dataclass, field
import logging
import os
from typing import List, Tuple
import torch


@dataclass
class ImageConfig:
    """Constraints and validation rules for uploaded images."""
    allowed_formats: List[str] = field(default_factory=lambda: ["JPEG", "JPG", "PNG", "WEBP"])
    max_file_size_bytes: int = 15 * 1024 * 1024  # 15 MB
    max_dimensions: Tuple[int, int] = (4096, 4096)
    safe_pixel_limit: int = 25_000_000  # Protection against decompression bombs
    max_inference_dimension: int = 1024  # Resize maximum dimension before heavy operations to prevent OOM


@dataclass
class BlurConfig:
    """Configuration for blur estimation and quality scoring."""
    laplacian_threshold: float = 100.0
    fft_high_freq_ratio_threshold: float = 0.02
    tenengrad_threshold: float = 300.0


@dataclass
class CaptioningConfig:
    """Inference parameters for BLIP caption generation."""
    model_id: str = "Salesforce/blip-image-captioning-base"
    
    # Concise mode parameters
    concise_num_beams: int = 3
    concise_max_new_tokens: int = 30
    concise_min_length: int = 5
    concise_repetition_penalty: float = 1.2
    
    # Detailed mode parameters
    detailed_prompt: str = "a photography showing"
    detailed_num_beams: int = 5
    detailed_max_new_tokens: int = 65
    detailed_min_length: int = 15
    detailed_repetition_penalty: float = 1.35
    
    # Reranking parameters
    num_candidates: int = 3


@dataclass
class RestorationConfig:
    """Settings for image deblurring and enhancement."""
    default_method: str = "adaptive"  # Options: 'adaptive', 'richardson_lucy', 'wiener', 'guided'
    max_restoration_dim: int = 1024
    rl_iterations: int = 12
    wiener_noise_ratio: float = 0.01


@dataclass
class AppConfig:
    """Global application configuration."""
    image: ImageConfig = field(default_factory=ImageConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    caption: CaptioningConfig = field(default_factory=CaptioningConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_level: int = logging.INFO


def get_logger(name: str = "ImageCaptionApp") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
