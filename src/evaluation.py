"""Comprehensive evaluation suite for image captioning and blur restoration."""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from src.config import get_logger
from src.preprocessing import cv2_to_pil, pil_to_cv2

logger = get_logger("Evaluation")


def tokenize_text(text: str) -> List[str]:
    """Cleans and tokenizes text into lowercased words."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return [tok for tok in cleaned.split() if tok]


def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    """Extracts n-gram counts from token list."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu_n(
    candidate: str,
    references: List[str],
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Computes standard BLEU-1, BLEU-2, BLEU-3, BLEU-4 with brevity penalty.
    References can be one or multiple ground-truth captions.
    """
    cand_tokens = tokenize_text(candidate)
    ref_token_lists = [tokenize_text(r) for r in references if r.strip()]

    if not cand_tokens or not ref_token_lists:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    # Effective reference length (closest length)
    c_len = len(cand_tokens)
    ref_lens = [len(r) for r in ref_token_lists]
    closest_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - c_len), r_len))

    # Brevity penalty
    if c_len > closest_ref_len:
        bp = 1.0
    elif c_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / c_len))

    cand_counts = [compute_ngram_counts(cand_tokens, n) for n in range(1, max_n + 1)]
    precisions = []

    for n in range(1, max_n + 1):
        c_n_counts = cand_counts[n - 1]
        total_cand_ngrams = sum(c_n_counts.values())

        if total_cand_ngrams == 0:
            precisions.append(0.0)
            continue

        # Max clipped counts across references
        clipped_matches = 0
        for ngram, count in c_n_counts.items():
            max_ref_count = max((compute_ngram_counts(r, n)[ngram] for r in ref_token_lists), default=0)
            clipped_matches += min(count, max_ref_count)

        precisions.append(clipped_matches / total_cand_ngrams)

    results = {}
    for i in range(1, max_n + 1):
        p_slice = precisions[:i]
        if any(p <= 0.0 for p in p_slice):
            results[f"bleu_{i}"] = 0.0
        else:
            w = 1.0 / i
            log_sum = sum(w * math.log(p) for p in p_slice)
            results[f"bleu_{i}"] = round(float(bp * math.exp(log_sum)), 4)

    return results


def compute_rouge_l(candidate: str, references: List[str]) -> float:
    """
    Computes ROUGE-L (Longest Common Subsequence F1 score) against reference captions.
    """
    cand_tokens = tokenize_text(candidate)
    if not cand_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        ref_tokens = tokenize_text(ref)
        if not ref_tokens:
            continue

        # Dynamic programming LCS
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if ref_tokens[i] == cand_tokens[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

        lcs = dp[m][n]
        if lcs == 0:
            continue

        p = lcs / n
        r = lcs / m
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1

    return round(float(best_f1), 4)


def compute_token_f1_score(candidate: str, references: List[str]) -> float:
    """Computes harmonic mean F1 of unigram tokens between candidate and references."""
    cand_tokens = set(tokenize_text(candidate))
    if not cand_tokens:
        return 0.0

    all_ref_tokens = set()
    for ref in references:
        all_ref_tokens.update(tokenize_text(ref))

    intersection = cand_tokens.intersection(all_ref_tokens)
    if not intersection:
        return 0.0

    precision = len(intersection) / len(cand_tokens)
    recall = len(intersection) / len(all_ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(float(f1), 4)


def evaluate_caption_quality(
    candidate: str,
    references: List[str]
) -> Dict[str, float]:
    """Calculates composite caption quality metrics."""
    bleu_metrics = compute_bleu_n(candidate, references, max_n=4)
    rouge_l = compute_rouge_l(candidate, references)
    token_f1 = compute_token_f1_score(candidate, references)

    return {
        **bleu_metrics,
        "rouge_l": rouge_l,
        "token_f1": token_f1,
    }


def evaluate_image_restoration_quality(
    clean_image: Union[Image.Image, np.ndarray],
    restored_image: Union[Image.Image, np.ndarray]
) -> Dict[str, float]:
    """
    Computes full-reference restoration metrics: PSNR (dB) and SSIM (0 to 1).
    Requires clean ground-truth reference image.
    """
    arr_clean = pil_to_cv2(clean_image) if isinstance(clean_image, Image.Image) else clean_image
    arr_restored = pil_to_cv2(restored_image) if isinstance(restored_image, Image.Image) else restored_image

    # Ensure identical dimensions
    if arr_clean.shape != arr_restored.shape:
        arr_restored = cv2.resize(
            arr_restored,
            (arr_clean.shape[1], arr_clean.shape[0]),
            interpolation=cv2.INTER_LANCZOS4
        )

    psnr_val = compute_psnr(arr_clean, arr_restored, data_range=255)
    ssim_val = compute_ssim(arr_clean, arr_restored, channel_axis=2, data_range=255)

    return {
        "psnr_db": round(float(psnr_val), 2),
        "ssim": round(float(ssim_val), 4),
    }
