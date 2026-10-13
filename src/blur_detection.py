"""Multi-metric blur detection and optical quality assessment."""

from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from PIL import Image

from src.config import BlurConfig, get_logger
from src.preprocessing import pil_to_cv2

logger = get_logger("BlurDetection")


@dataclass
class BlurAssessmentResult:
    """Detailed output of image blur and clarity estimation."""
    is_blurry: bool
    blur_level: str  # 'sharp', 'mild_blur', 'severe_blur'
    clarity_score: float  # 0.0 (extremely blurry) to 100.0 (crystal sharp)
    laplacian_variance: float
    fft_high_freq_ratio: float
    tenengrad_score: float
    estimated_blur_type: str  # 'none', 'motion_blur', 'defocus_blur', 'low_texture'
    recommendation: str


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Computes second derivative variance (Pech-Pacheco et al.)."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_fft_high_frequency_ratio(gray: np.ndarray, radius_ratio: float = 0.15) -> float:
    """
    Computes ratio of high-frequency energy in 2D Discrete Fourier Transform.
    Blurry images lose high-frequency power.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    # 2D FFT and shift zero frequency to center
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    cy, cx = h // 2, w // 2
    r = int(min(cy, cx) * radius_ratio)

    # Low frequency mask
    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    total_energy = np.sum(magnitude_spectrum) + 1e-8
    low_freq_energy = np.sum(magnitude_spectrum[mask_low])
    high_freq_energy = total_energy - low_freq_energy

    return float(high_freq_energy / total_energy)


def compute_tenengrad_score(gray: np.ndarray) -> float:
    """Computes Tenengrad gradient energy using Sobel operators."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(gradient_magnitude**2))


def estimate_directional_blur(gray: np.ndarray) -> str:
    """
    Analyzes horizontal vs vertical gradient variances to determine
    if blur is directional (motion blur) or isotropic (defocus blur).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    var_x = float(np.var(gx))
    var_y = float(np.var(gy))
    
    if var_x < 1e-6 and var_y < 1e-6:
        return "none"
        
    ratio = max(var_x, var_y) / (min(var_x, var_y) + 1e-5)
    if ratio > 2.2:
        return "motion_blur"
    return "defocus_blur"


def assess_image_blur(
    image: Image.Image,
    config: Optional[BlurConfig] = None
) -> BlurAssessmentResult:
    """
    Comprehensive multi-metric blur assessment combining Laplacian variance,
    Fourier frequency analysis, and Tenengrad gradient statistics.
    """
    if config is None:
        config = BlurConfig()

    arr = pil_to_cv2(image)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    lap_var = compute_laplacian_variance(gray)
    fft_ratio = compute_fft_high_frequency_ratio(gray)
    tenengrad = compute_tenengrad_score(gray)

    # Calculate normalized clarity score (0 to 100)
    # Using smooth logarithmic/sigmoid scaling calibrated against natural photos
    lap_component = min(100.0, (np.log1p(lap_var) / np.log1p(1000.0)) * 100.0)
    fft_component = min(100.0, (fft_ratio / 0.10) * 100.0)
    ten_component = min(100.0, (np.log1p(tenengrad) / np.log1p(2000.0)) * 100.0)
    
    clarity_score = float(np.clip(
        0.50 * lap_component + 0.30 * fft_component + 0.20 * ten_component,
        0.0,
        100.0
    ))

    # Determine blur level and degradation type
    if lap_var < 35.0 or clarity_score < 35.0:
        blur_level = "severe_blur"
        is_blurry = True
        estimated_type = estimate_directional_blur(gray)
        recommendation = (
            "Image is heavily degraded. Deblurring may improve salient structure, "
            "but severe blur permanently destroys fine detail."
        )
    elif lap_var < config.laplacian_threshold or clarity_score < 55.0:
        blur_level = "mild_blur"
        is_blurry = True
        estimated_type = estimate_directional_blur(gray)
        recommendation = "Image exhibits moderate softness or blur. Enhancement is recommended."
    else:
        blur_level = "sharp"
        is_blurry = False
        estimated_type = "none"
        recommendation = "Image is clear and sharp. Enhancement is unnecessary."

    logger.info(
        f"Blur assessment: level={blur_level}, score={clarity_score:.1f}/100, "
        f"laplacian={lap_var:.2f}, fft_ratio={fft_ratio:.4f}, type={estimated_type}"
    )

    return BlurAssessmentResult(
        is_blurry=is_blurry,
        blur_level=blur_level,
        clarity_score=round(clarity_score, 1),
        laplacian_variance=round(lap_var, 2),
        fft_high_freq_ratio=round(fft_ratio, 4),
        tenengrad_score=round(tenengrad, 2),
        estimated_blur_type=estimated_type,
        recommendation=recommendation,
    )
