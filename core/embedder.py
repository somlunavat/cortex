"""Local embedding model wrapper with lazy loading and precision quantization.

Uses sentence-transformers (all-MiniLM-L6-v2) as primary backend.
Falls back gracefully if the model is unavailable (returns zero vector).
No external API calls. Model loads once at first use (singleton).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384
_MODEL_NAME = "all-MiniLM-L6-v2"

_model: object = None
_model_loaded: bool = False


def _get_model() -> object | None:
    """Lazy-load the sentence-transformers model. Returns None on failure."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
        logger.debug("Loaded embedding model %s", _MODEL_NAME)
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Could not load embedding model: %s; using zero vectors", exc)
        _model = None
    return _model


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of strings in a single model forward pass.

    Much faster than calling embed() in a loop because sentence-transformers
    can batch-encode on GPU or optimised CPU kernels. Falls back to zero
    vectors if the model is unavailable.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float32 numpy arrays of shape (384,), one per input string.
    """
    if not texts:
        return []

    model = _get_model()
    if model is None:
        return [np.zeros(_EMBEDDING_DIM, dtype=np.float32) for _ in texts]

    try:
        from sentence_transformers import SentenceTransformer

        assert isinstance(model, SentenceTransformer)
        results = model.encode(texts, convert_to_numpy=True, batch_size=64)
        out: list[np.ndarray] = []
        for r in results:
            arr = np.asarray(r, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.squeeze().astype(np.float32)
            out.append(arr)
        return out
    except (RuntimeError, ValueError, AssertionError) as exc:
        logger.warning("Batch embedding failed: %s; using zero vectors", exc)
        return [np.zeros(_EMBEDDING_DIM, dtype=np.float32) for _ in texts]


def embed(text: str) -> np.ndarray:
    """Embed a text string into a 384-dimensional float32 vector.

    Loads the model on first call. Returns a zero vector if the model
    fails to load.

    Args:
        text: Input text to embed (typically one sentence).

    Returns:
        numpy float32 array of shape (384,).
    """
    model = _get_model()
    if model is None:
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)
    try:
        from sentence_transformers import SentenceTransformer

        assert isinstance(model, SentenceTransformer)
        result = model.encode(text, convert_to_numpy=True)
        arr = np.asarray(result, dtype=np.float32)
        if arr.ndim > 1:
            return arr.squeeze().astype(np.float32)
        return arr
    except (RuntimeError, ValueError, AssertionError) as exc:
        logger.warning("Embedding failed for text %r: %s", text[:40], exc)
        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)


def serialize(embedding: np.ndarray, precision_bits: int = 32) -> bytes:
    """Serialize a numpy embedding array to bytes for SQLite BLOB storage.

    Applies precision quantization based on tier:
    - 32 bits: float32 (lossless)
    - 8 bits: int8 quantization (scaled to [-127, 127])
    - 2 bits: int2 quantization (clipped to {-1, 0, 1} stored as int8)

    Args:
        embedding: Float32 numpy array to serialize.
        precision_bits: Target precision (32, 8, or 2).

    Returns:
        Raw bytes suitable for SQLite BLOB storage.

    Raises:
        ValueError: If precision_bits is not 32, 8, or 2.
    """
    arr = embedding.astype(np.float32)
    match precision_bits:
        case 32:
            return arr.tobytes()
        case 8:
            quantized = _quantize_int8(arr)
            return quantized.tobytes()
        case 2:
            quantized = _quantize_int2(arr)
            return quantized.tobytes()
        case _:
            raise ValueError(
                f"Unsupported precision_bits: {precision_bits}. Must be 32, 8, or 2."
            )


def deserialize(blob: bytes, precision_bits: int = 32) -> np.ndarray:
    """Deserialize bytes from SQLite BLOB back to a float32 numpy array.

    Reverses the quantization applied during serialize(). The returned array
    is always float32 regardless of storage precision.

    Args:
        blob: Raw bytes from SQLite BLOB column.
        precision_bits: Precision used when the blob was serialized.

    Returns:
        Float32 numpy array of shape (384,).

    Raises:
        ValueError: If precision_bits is not 32, 8, or 2.
    """
    match precision_bits:
        case 32:
            return np.frombuffer(blob, dtype=np.float32).copy()
        case 8:
            quantized = np.frombuffer(blob, dtype=np.int8).copy()
            return _dequantize_int8(quantized)
        case 2:
            quantized = np.frombuffer(blob, dtype=np.int8).copy()
            return quantized.astype(np.float32)
        case _:
            raise ValueError(
                f"Unsupported precision_bits: {precision_bits}. Must be 32, 8, or 2."
            )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        a: Numpy array (any float dtype; converted to float32 if needed).
        b: Numpy array (any float dtype; converted to float32 if needed).

    Returns:
        Cosine similarity in the range [-1.0, 1.0]. Returns 0.0 if either
        vector has zero norm.
    """
    a_f = a if a.dtype == np.float32 else a.astype(np.float32)
    b_f = b if b.dtype == np.float32 else b.astype(np.float32)
    norm_a = float(np.linalg.norm(a_f))
    norm_b = float(np.linalg.norm(b_f))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_f, b_f) / (norm_a * norm_b))


def cosine_similarity_batch(
    query: np.ndarray,
    embeddings: list[np.ndarray],
) -> list[float]:
    """Compute cosine similarity between a query and a list of embeddings.

    Stacks all embeddings into a matrix and computes dot products in a single
    BLAS call — significantly faster than calling cosine_similarity() in a
    loop when the candidate list is large.

    Args:
        query: Float32 query vector of shape (D,).
        embeddings: List of float32 vectors, each of shape (D,).

    Returns:
        List of cosine similarities in the same order as embeddings. Vectors
        with zero norm receive a score of 0.0.
    """
    if not embeddings:
        return []

    q = query if query.dtype == np.float32 else query.astype(np.float32)
    norm_q = float(np.linalg.norm(q))
    if norm_q == 0.0:
        return [0.0] * len(embeddings)

    mat = np.stack(
        [e if e.dtype == np.float32 else e.astype(np.float32) for e in embeddings]
    )
    norms = np.linalg.norm(mat, axis=1)
    dots = mat @ q
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = np.where(norms > 0.0, dots / (norms * norm_q), 0.0)

    return sims.tolist()


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _quantize_int8(arr: np.ndarray) -> np.ndarray:
    """Quantize a float32 array to int8 range [-127, 127]."""
    max_val = float(np.max(np.abs(arr)))
    if max_val == 0.0:
        return np.zeros_like(arr, dtype=np.int8)
    scaled = arr * (127.0 / max_val)
    return np.clip(np.round(scaled), -127, 127).astype(np.int8)


def _dequantize_int8(arr: np.ndarray) -> np.ndarray:
    """Dequantize an int8 array back to float32 (approximate, not lossless)."""
    return arr.astype(np.float32) / 127.0


def _quantize_int2(arr: np.ndarray) -> np.ndarray:
    """Quantize to ternary values {-1, 0, 1} stored as int8.

    Values above the mean absolute value → 1
    Values below the negated mean absolute value → -1
    Others → 0
    """
    threshold = float(np.mean(np.abs(arr)))
    result = np.zeros_like(arr, dtype=np.int8)
    result[arr > threshold] = 1
    result[arr < -threshold] = -1
    return result


def embedding_dim() -> int:
    """Return the expected embedding dimension.

    Returns:
        Integer dimension of all embeddings produced by this module.
    """
    return _EMBEDDING_DIM
