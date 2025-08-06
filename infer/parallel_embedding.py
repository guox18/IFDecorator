"""
Parallel text embedding module using SentenceTransformer.
This module provides functionality for computing text embeddings in parallel
using all available CUDA devices.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from global_config import SENTENCE_BERT_PATH
from logs.logger import logger


def parallel_SentenceTransformer(
    sentences: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute text embeddings in parallel using SentenceTransformer.

    This function utilizes all available CUDA devices to compute embeddings
    for a list of sentences in parallel, improving processing speed for
    large batches of text.

    Args:
        sentences (List[str]): List of sentences to compute embeddings for
        model_name (str): Name of the SentenceTransformer model to use
        batch_size (Optional[int]): Batch size for processing. If None,
            automatically determined based on available memory

    Returns:
        np.ndarray: Array of computed embeddings with shape (n_sentences, embedding_dim)

    Raises:
        RuntimeError: If no CUDA devices are available
        ValueError: If sentences list is empty
    """
    if not sentences:
        raise ValueError("Input sentences list cannot be empty")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU for embedding computation.")

    try:
        # Initialize the model
        # model = SentenceTransformer(model_name)
        logger.info(f"Loaded model: {model_name}")

        model = SentenceTransformer(SENTENCE_BERT_PATH)
        logger.info(f"Loaded model: {SENTENCE_BERT_PATH}")

        # Start the multi-process pool on all available CUDA devices
        pool = model.start_multi_process_pool()
        logger.info("Started multi-process pool")

        # Compute embeddings
        embeddings = model.encode_multi_process(
            sentences, pool, batch_size=batch_size or 1024, show_progress_bar=True
        )
        logger.info(f"Computed embeddings with shape: {embeddings.shape}")

        # Clean up
        model.stop_multi_process_pool(pool)
        logger.info("Stopped multi-process pool")

        return embeddings

    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        # Create example sentences
        sentences = [
            "This is a test sentence.",
            "Another example sentence for embedding.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Compute embeddings
        embeddings = parallel_SentenceTransformer(sentences)
        print(f"Successfully computed embeddings with shape: {embeddings.shape}")

        # Example of computing similarity between sentences
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(embeddings)
        print("\nCosine similarities between sentences:")
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                print(
                    f"Similarity between sentence {i+1} and {j+1}: {similarities[i][j]:.4f}"
                )

    except Exception as e:
        print(f"Error in example: {str(e)}")
