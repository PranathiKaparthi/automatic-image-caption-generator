"""Unit tests for captioning logic, formatting, and mocked model responses."""

from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
import torch

from src.captioning import CaptionGenerator, CaptionResult
from src.config import CaptioningConfig


def test_postprocess_caption_formatting():
    """Verifies that generated strings are capitalized and properly punctuated."""
    generator = CaptionGenerator()
    assert generator._postprocess_caption("a black cat on a rug") == "A black cat on a rug."
    assert generator._postprocess_caption("a white dog.") == "A white dog."
    assert generator._postprocess_caption("") == "An unidentifiable scene."


def test_mocked_caption_inference():
    """Verifies generation pipeline handling mocked processor and model."""
    generator = CaptionGenerator()
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))

    # Mock processor and model
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
    mock_processor.decode.return_value = "a sunny day at the beach"

    mock_outputs = MagicMock()
    mock_outputs.sequences = torch.zeros((1, 10), dtype=torch.long)
    mock_model.generate.return_value = mock_outputs

    generator.processor = mock_processor
    generator.model = mock_model
    generator.is_loaded = True

    result = generator.generate_caption(img, mode="concise", num_candidates=1)

    assert result.error is None
    assert result.caption == "A sunny day at the beach."
    assert result.num_tokens == 10
    assert result.mode == "concise"


def test_caption_error_handling():
    """Verifies that internal errors return clean error objects instead of crashing."""
    generator = CaptionGenerator()
    img = Image.new("RGB", (100, 100))

    generator.processor = MagicMock(side_effect=RuntimeError("CUDA out of memory mock"))
    generator.model = MagicMock()
    generator.is_loaded = True

    result = generator.generate_caption(img)
    assert result.error is not None
    assert "CUDA out of memory mock" in result.error
    assert result.caption == ""
