"""
Hybrid defense
"""

# pylint: disable=W0102

import time
from collections.abc import Sequence
from pathlib import Path

import joblib
from defense_utils import EnhancedPromptInjectionDetector, push_contextarg
from langchain_huggingface import HuggingFaceEmbeddings

from agentdojo.logging import TraceLogger
from agentdojo.types import ChatMessage


class EmbeddingPIDetector(EnhancedPromptInjectionDetector):
    """
    Uses a random forest classifier with an embedding model to perform preliminary classification,
    deferring uncertain cases to an LLM-based classifier.

    Args:
        - `embedding_model_name`: The name of the embedding model to use for the preliminary
            classification.
        - `embedding_classifier_path`: The path to the joblib of the trained embedding injection
            classifier.
        - `embedding_classifier_min_benign`: The minimum probability for the embedding classifier
            to classify an input as benign, skipping the LLM-based secondary classification.
        - `embedding_classifier_min_malicious`: The minimum probability for the embedding classifier
            to classify an input as malicious, skipping the LLM-based secondary classification.
    """

    def __init__(
        self,
        embedding_model_name: str,
        embedding_classifier_path: Path,
        embedding_classifier_min_benign: float = 0.5,
        embedding_classifier_min_malicious: float = 0.5,
    ) -> None:
        super().__init__()

        # Initialize the state
        self.embedding_classifier_min_benign = embedding_classifier_min_benign
        self.embedding_classifier_min_malicious = embedding_classifier_min_malicious

        # Initialize the embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            show_progress=False,
        )

        # Initialize the pre-trained classifier
        self.classifier = joblib.load(embedding_classifier_path)
        self.classifier.verbose = 0

    def detect(
        self, tool_output: str, messages: Sequence[ChatMessage], logger: TraceLogger
    ) -> tuple[bool, float]:
        # Perform the preliminary classification
        preliminary_start = time.monotonic()
        embeddings = self.embedding_model.embed_documents(texts=[tool_output])[0]
        preliminary_prediction = self.classifier.predict_proba([embeddings])[0]
        is_benign = preliminary_prediction[0] >= self.embedding_classifier_min_benign
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

        return is_benign, preliminary_prediction[1]
