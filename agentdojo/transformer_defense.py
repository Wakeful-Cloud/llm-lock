"""
Transformer-only defense
"""

# pylint: disable=W0102

import time
from collections.abc import Sequence
from typing import Any

import torch
from defense_utils import EnhancedPromptInjectionDetector, push_contextarg
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from agentdojo.logging import TraceLogger
from agentdojo.types import ChatMessage


class TransformerPIDetector(EnhancedPromptInjectionDetector):
    """
    Uses an LLM-based classifier to directly detect prompt injections.

    Args:
        - `transformer_model_name`: The name of the transformer model to use for prompt injection
            detection.
        - `transformer_benign_label`: The label that indicates a benign prompt.
        - `transformer_threshold`: The threshold for the transformer model's prediction to be
            considered a prompt injection (e.g., malicious probability >= `transformer_threshold`
            means an input is classified as a prompt injection).
    """

    def __init__(
        self,
        transformer_model_name: str,
        transformer_benign_label: str,
        transformer_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        # Initialize the state
        self.transformer_benign_label = transformer_benign_label
        self.transformer_threshold = transformer_threshold

        # Initialize the LLM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        llm_model = AutoModelForSequenceClassification.from_pretrained(
            transformer_model_name, dtype=dtype
        )

        if llm_model.config.id2label[0] != transformer_benign_label:
            raise ValueError(
                f"Model's benign label {llm_model.config.id2label[0]} does not match expected benign label {transformer_benign_label} (Are you sure you updated the transformer_benign_label parameter?)"
            )

        # Initialize the classification pipeline
        self.pipeline = pipeline(
            "text-classification",
            model=llm_model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            device=device,
        )

    def detect(
        self, tool_output: str, messages: Sequence[ChatMessage], logger: TraceLogger
    ) -> tuple[bool, float]:
        # Filter out empty tool outputs
        if tool_output == "":
            return False, 0.0

        # Run through the LLM classifier
        start = time.monotonic()
        result: dict[str, Any] = self.pipeline(tool_output)[0]
        malicious_probability: float = (
            result["score"]
            if result["label"] != self.transformer_benign_label
            else 1 - result["score"]
        )
        is_injection = malicious_probability >= self.transformer_threshold
        end = time.monotonic()

        # Log the context
        push_contextarg(logger, "classification_duration", end - start)
        push_contextarg(
            logger, "classification_malicious_probability", malicious_probability
        )

        return is_injection, malicious_probability
