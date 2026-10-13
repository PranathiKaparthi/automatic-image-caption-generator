"""Image captioning engine utilizing pretrained vision-language models."""

from dataclasses import dataclass, field
import time
from typing import List, Optional
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import CaptioningConfig, get_logger
from src.preprocessing import safe_resize_for_inference

logger = get_logger("Captioning")


@dataclass
class CaptionResult:
    """Structured output of caption inference."""
    caption: str
    candidates: List[str] = field(default_factory=list)
    inference_time_s: float = 0.0
    mode: str = "concise"
    model_name: str = ""
    device: str = "cpu"
    num_tokens: int = 0
    error: Optional[str] = None


class CaptionGenerator:
    """
    Singleton-capable manager for loading and executing vision-language captioning models.
    Loads model weights once into memory and reuses them across all subsequent requests.
    """
    _instance: Optional["CaptionGenerator"] = None

    def __init__(self, config: Optional[CaptioningConfig] = None, device: Optional[str] = None):
        self.config = config or CaptioningConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        self.is_loaded: bool = False
        self.cold_start_time_s: float = 0.0

    @classmethod
    def get_instance(cls, config: Optional[CaptioningConfig] = None) -> "CaptionGenerator":
        """Returns or creates the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def load_model(self) -> float:
        """
        Loads processor and model weights once.
        Returns cold-start loading time in seconds.
        """
        if self.is_loaded and self.model is not None and self.processor is not None:
            logger.info("Model already in memory, skipping reload.")
            return 0.0

        t0 = time.time()
        logger.info(f"Loading captioning model '{self.config.model_id}' on device '{self.device}'...")
        
        try:
            self.processor = BlipProcessor.from_pretrained(self.config.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.config.model_id)
            self.model.to(self.device)
            self.model.eval()
            
            # Disable gradients permanently for inference
            for param in self.model.parameters():
                param.requires_grad = False

            self.cold_start_time_s = round(time.time() - t0, 3)
            self.is_loaded = True
            logger.info(f"Model loaded successfully in {self.cold_start_time_s:.2f}s.")
            return self.cold_start_time_s

        except Exception as e:
            logger.error(f"Failed to load captioning model: {e}", exc_info=True)
            self.is_loaded = False
            raise RuntimeError(f"Could not load captioning model '{self.config.model_id}': {str(e)}")

    def _postprocess_caption(self, raw_caption: str) -> str:
        """Cleans, formats, and ensures grammatical capitalization."""
        cap = raw_caption.strip()
        if not cap:
            return "An unidentifiable scene."
        # Capitalize first letter
        cap = cap[0].upper() + cap[1:]
        # Ensure ends with punctuation
        if not cap.endswith((".", "!", "?")):
            cap += "."
        return cap

    def generate_caption(
        self,
        image: Image.Image,
        mode: str = "concise",
        num_candidates: int = 1,
        rerank: bool = False,
        custom_prompt: Optional[str] = None
    ) -> CaptionResult:
        """
        Generates descriptive caption for the provided image.
        
        Args:
            image: Normalized PIL RGB image.
            mode: 'concise' (short, factual) or 'detailed' (rich contextual).
            num_candidates: Number of candidate captions to generate with beam search.
            rerank: Whether to evaluate and score multiple candidates.
            custom_prompt: Optional conditional text prompt.
        """
        if not self.is_loaded:
            self.load_model()

        assert self.model is not None
        assert self.processor is not None

        t0 = time.time()
        try:
            # Safe resize if exceptionally large to ensure predictable latency
            prep_img, _ = safe_resize_for_inference(image, max_dimension=1024)

            # Determine generation hyperparameters based on mode
            if mode == "detailed":
                prompt = custom_prompt or self.config.detailed_prompt
                max_new_tokens = self.config.detailed_max_new_tokens
                min_length = self.config.detailed_min_length
                num_beams = max(self.config.detailed_num_beams, num_candidates)
                rep_penalty = self.config.detailed_repetition_penalty
            else:
                prompt = custom_prompt  # None by default for unconditional captioning
                max_new_tokens = self.config.concise_max_new_tokens
                min_length = self.config.concise_min_length
                num_beams = max(self.config.concise_num_beams, num_candidates)
                rep_penalty = self.config.concise_repetition_penalty

            # Tokenize & preprocess visual inputs
            if prompt:
                inputs = self.processor(images=prep_img, text=prompt, return_tensors="pt")
            else:
                inputs = self.processor(images=prep_img, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=num_beams,
                    num_return_sequences=num_candidates,
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    repetition_penalty=rep_penalty,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True if (rerank and num_candidates > 1) else False,
                )

            # Decode sequences
            decoded_candidates: List[str] = []
            for seq in outputs.sequences:
                text = self.processor.decode(seq, skip_special_tokens=True)
                # Strip leading prompt if prompt was provided
                if prompt and text.lower().startswith(prompt.lower()):
                    text = text[len(prompt):].strip()
                cleaned = self._postprocess_caption(text)
                decoded_candidates.append(cleaned)

            # Deduplicate preserving order
            unique_candidates = list(dict.fromkeys(decoded_candidates))

            # Select best candidate (with reranking if requested)
            if rerank and len(unique_candidates) > 1 and hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                # Rank by sequence score length-normalized
                best_idx = int(torch.argmax(outputs.sequences_scores).item())
                best_caption = decoded_candidates[best_idx]
            else:
                best_caption = unique_candidates[0] if unique_candidates else "A scene."

            t_elapsed = time.time() - t0
            num_tokens = int(outputs.sequences.shape[1]) if hasattr(outputs, "sequences") else 0

            logger.info(
                f"Generated [{mode}] caption in {t_elapsed:.3f}s: '{best_caption}' "
                f"({num_tokens} tokens)"
            )

            return CaptionResult(
                caption=best_caption,
                candidates=unique_candidates,
                inference_time_s=round(t_elapsed, 3),
                mode=mode,
                model_name=self.config.model_id,
                device=self.device,
                num_tokens=num_tokens,
                error=None,
            )

        except Exception as e:
            logger.error(f"Inference error during caption generation: {e}", exc_info=True)
            return CaptionResult(
                caption="",
                candidates=[],
                inference_time_s=round(time.time() - t0, 3),
                mode=mode,
                model_name=self.config.model_id,
                device=self.device,
                num_tokens=0,
                error=f"Inference failed: {str(e)}",
            )
