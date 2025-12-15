"""
Hybrid defense
"""

# pylint: disable=W0102

import time
from collections.abc import Sequence
from pathlib import Path

import joblib
import torch
from defense_utils import EnhancedPromptInjectionDetector, push_contextarg
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from agentdojo.logging import TraceLogger
from agentdojo.types import ChatMessage


class HybridPIDetector(EnhancedPromptInjectionDetector):
    """
    Uses a random forest classifier with an embedding model to perform preliminary classification,
    deferring uncertain cases to an LLM-based classifier.

    Args:
        - `embedding_model_name`: The name of the embedding model to use for the preliminary
            classification.
        - `embedding_classifier_path`: The path to the joblib of the trained embedding injection
            classifier.
        - `transformer_model_name`: The name of the transformer model to use for secondary prompt
            injection detection.
        - `transformer_benign_label`: The label that indicates a benign prompt.
        - `embedding_classifier_min_benign`: The minimum probability for the embedding classifier
            to classify an input as benign, skipping the LLM-based secondary classification.
        - `embedding_classifier_min_malicious`: The minimum probability for the embedding classifier
            to classify an input as malicious, skipping the LLM-based secondary classification.
        - `transformer_threshold`: The threshold for the transformer model's prediction to be
            considered a prompt injection (e.g., malicious probability >= `transformer_threshold`
            means an input is classified as a prompt injection).
    """

    def __init__(
        self,
        embedding_model_name: str,
        embedding_classifier_path: Path,
        transformer_model_name: str,
        transformer_benign_label: str,
        embedding_classifier_min_benign: float = 0.6,
        embedding_classifier_min_malicious: float = 0.6,
        transformer_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        # Initialize the state
        self.transformer_benign_label = transformer_benign_label
        self.embedding_classifier_min_benign = embedding_classifier_min_benign
        self.embedding_classifier_min_malicious = embedding_classifier_min_malicious
        self.transformer_threshold = transformer_threshold

        # Initialize the embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            show_progress=False,
        )

        # Initialize the pre-trained classifier
        self.classifier = joblib.load(embedding_classifier_path)
        self.classifier.verbose = 0

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
        # Perform the preliminary classification
        preliminary_start = time.monotonic()
        embeddings = self.embedding_model.embed_documents(texts=[tool_output])[0]
        preliminary_prediction = self.classifier.predict_proba([embeddings])[0]
        is_benign = preliminary_prediction[0] >= self.embedding_classifier_min_benign
        is_malicious = (
            preliminary_prediction[1] >= self.embedding_classifier_min_malicious
        )
        preliminary_end = time.monotonic()

        # Log the context
        push_contextarg(
            logger,
            "preliminary_classification_duration",
            preliminary_end - preliminary_start,
        )
        push_contextarg(
            logger,
            "preliminary_classification_malicious_probability",
            preliminary_prediction[1],
        )

        # If high confidence, return early
        if is_benign:
            return False, preliminary_prediction[1]
        elif is_malicious:
            return True, preliminary_prediction[1]

        # Perform the secondary classification
        secondary_start = time.monotonic()
        result: dict = self.pipeline(tool_output)[0]
        secondary_malicious_probability: float = (
            result["score"]
            if result["label"] != self.transformer_benign_label
            else 1 - result["score"]
        )
        is_injection: bool = (
            secondary_malicious_probability >= self.transformer_threshold
        )
        secondary_end = time.monotonic()

        # Log the context
        push_contextarg(
            logger, "secondary_classification_duration", secondary_end - secondary_start
        )
        push_contextarg(
            logger,
            "secondary_classification_malicious_probability",
            secondary_malicious_probability,
        )

        return is_injection, secondary_malicious_probability
