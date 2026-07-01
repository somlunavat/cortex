"""Tests for core/embedder.py — local embedding wrapper with precision quantization."""

from __future__ import annotations

import numpy as np
import pytest

from core.embedder import (
    _EMBEDDING_DIM,
    _dequantize_int8,
    _quantize_int2,
    _quantize_int8,
    cosine_similarity,
    deserialize,
    embed,
    embed_batch,
    embedding_dim,
    serialize,
)


@pytest.fixture
def dummy_vec() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.random(_EMBEDDING_DIM).astype(np.float32)


@pytest.fixture
def zero_vec() -> np.ndarray:
    return np.zeros(_EMBEDDING_DIM, dtype=np.float32)


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    def test_returns_list_of_correct_length(self) -> None:
        texts = ["hello world", "asyncpg is fast", "JWT auth"]
        result = embed_batch(texts)
        assert len(result) == len(texts)

    def test_each_result_is_384_dim(self) -> None:
        result = embed_batch(["a", "b"])
        for vec in result:
            assert vec.shape == (_EMBEDDING_DIM,)

    def test_each_result_is_float32(self) -> None:
        result = embed_batch(["a", "b"])
        for vec in result:
            assert vec.dtype == np.float32

    def test_empty_input_returns_empty_list(self) -> None:
        assert embed_batch([]) == []

    def test_batch_matches_individual_embed(self) -> None:
        texts = ["asyncpg is fast", "JWT authentication"]
        batch = embed_batch(texts)
        for text, batch_vec in zip(texts, batch, strict=True):
            single_vec = embed(text)
            np.testing.assert_allclose(single_vec, batch_vec, atol=1e-5)

    def test_single_item_batch(self) -> None:
        result = embed_batch(["hello"])
        assert len(result) == 1
        assert result[0].shape == (_EMBEDDING_DIM,)


class TestEmbed:
    def test_returns_384_dim_vector(self) -> None:
        vec = embed("hello world")
        assert vec.shape == (_EMBEDDING_DIM,)

    def test_returns_float32(self) -> None:
        vec = embed("hello world")
        assert vec.dtype == np.float32

    def test_different_texts_produce_different_vectors(self) -> None:
        v1 = embed("asyncpg is fast")
        v2 = embed("the cat sat on the mat")
        assert not np.allclose(v1, v2, atol=1e-4)

    def test_same_text_produces_same_vector(self) -> None:
        v1 = embed("I chose asyncpg")
        v2 = embed("I chose asyncpg")
        np.testing.assert_array_equal(v1, v2)

    def test_empty_string_returns_vector(self) -> None:
        vec = embed("")
        assert vec.shape == (_EMBEDDING_DIM,)

    def test_model_loads_once(self) -> None:
        from core.embedder import _get_model

        first = _get_model()
        second = _get_model()
        assert first is second


# ---------------------------------------------------------------------------
# serialize / deserialize — float32
# ---------------------------------------------------------------------------


class TestSerializeDeserializeFloat32:
    def test_round_trip_float32(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=32)
        recovered = deserialize(blob, precision_bits=32)
        np.testing.assert_allclose(recovered, dummy_vec, atol=1e-6)

    def test_blob_is_bytes(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=32)
        assert isinstance(blob, bytes)

    def test_float32_blob_size(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=32)
        assert len(blob) == _EMBEDDING_DIM * 4

    def test_invalid_precision_raises(self, dummy_vec: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unsupported precision_bits"):
            serialize(dummy_vec, precision_bits=16)

    def test_deserialize_invalid_precision_raises(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=32)
        with pytest.raises(ValueError, match="Unsupported precision_bits"):
            deserialize(blob, precision_bits=16)


# ---------------------------------------------------------------------------
# serialize / deserialize — int8
# ---------------------------------------------------------------------------


class TestSerializeDeserializeInt8:
    def test_round_trip_approximate_int8(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=8)
        recovered = deserialize(blob, precision_bits=8)
        assert recovered.dtype == np.float32
        assert recovered.shape == (_EMBEDDING_DIM,)

    def test_int8_blob_smaller_than_float32(self, dummy_vec: np.ndarray) -> None:
        blob32 = serialize(dummy_vec, precision_bits=32)
        blob8 = serialize(dummy_vec, precision_bits=8)
        assert len(blob8) < len(blob32)

    def test_int8_blob_size(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=8)
        assert len(blob) == _EMBEDDING_DIM

    def test_int8_values_in_range(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=8)
        arr = np.frombuffer(blob, dtype=np.int8)
        assert np.all(arr >= -127)
        assert np.all(arr <= 127)

    def test_int8_cosine_similarity_preserved(self, dummy_vec: np.ndarray) -> None:
        rng = np.random.default_rng(seed=99)
        other_vec = rng.random(_EMBEDDING_DIM).astype(np.float32)
        original_sim = cosine_similarity(dummy_vec, other_vec)
        blob = serialize(dummy_vec, precision_bits=8)
        recovered = deserialize(blob, precision_bits=8)
        approx_sim = cosine_similarity(recovered, other_vec)
        assert abs(original_sim - approx_sim) < 0.15


# ---------------------------------------------------------------------------
# serialize / deserialize — int2
# ---------------------------------------------------------------------------


class TestSerializeDeserializeInt2:
    def test_round_trip_int2(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=2)
        recovered = deserialize(blob, precision_bits=2)
        assert recovered.shape == (_EMBEDDING_DIM,)

    def test_int2_values_are_ternary(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=2)
        arr = np.frombuffer(blob, dtype=np.int8)
        unique_vals = set(arr.tolist())
        assert unique_vals <= {-1, 0, 1}

    def test_int2_blob_size(self, dummy_vec: np.ndarray) -> None:
        blob = serialize(dummy_vec, precision_bits=2)
        assert len(blob) == _EMBEDDING_DIM


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_1(self, dummy_vec: np.ndarray) -> None:
        sim = cosine_similarity(dummy_vec, dummy_vec)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_return_0(self) -> None:
        a = np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        b[1] = 1.0
        sim = cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector_returns_0(
        self, dummy_vec: np.ndarray, zero_vec: np.ndarray
    ) -> None:
        assert cosine_similarity(dummy_vec, zero_vec) == 0.0
        assert cosine_similarity(zero_vec, dummy_vec) == 0.0

    def test_both_zero_returns_0(self, zero_vec: np.ndarray) -> None:
        assert cosine_similarity(zero_vec, zero_vec) == 0.0

    def test_similarity_in_range(self, dummy_vec: np.ndarray) -> None:
        rng = np.random.default_rng(seed=7)
        other = rng.random(_EMBEDDING_DIM).astype(np.float32)
        sim = cosine_similarity(dummy_vec, other)
        assert -1.0 <= sim <= 1.0

    @pytest.mark.parametrize("scale", [0.5, 2.0, -1.0])
    def test_similarity_invariant_to_scale(
        self, dummy_vec: np.ndarray, scale: float
    ) -> None:
        rng = np.random.default_rng(seed=11)
        other = rng.random(_EMBEDDING_DIM).astype(np.float32)
        sim_orig = cosine_similarity(dummy_vec, other)
        sim_scaled = cosine_similarity(dummy_vec * scale, other)
        assert abs(sim_orig - abs(sim_scaled)) < 1e-4


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


class TestQuantizationHelpers:
    def test_quantize_int8_zero_array(self) -> None:
        arr = np.zeros(10, dtype=np.float32)
        result = _quantize_int8(arr)
        assert np.all(result == 0)

    def test_quantize_int8_max_magnitude(self, dummy_vec: np.ndarray) -> None:
        result = _quantize_int8(dummy_vec)
        assert int(np.max(np.abs(result))) == 127

    def test_dequantize_int8_produces_float32(self) -> None:
        arr = np.array([127, 0, -127], dtype=np.int8)
        result = _dequantize_int8(arr)
        assert result.dtype == np.float32

    def test_quantize_int2_only_ternary_values(self, dummy_vec: np.ndarray) -> None:
        result = _quantize_int2(dummy_vec)
        unique = set(result.tolist())
        assert unique <= {-1, 0, 1}

    def test_embedding_dim_returns_384(self) -> None:
        assert embedding_dim() == 384


# ---------------------------------------------------------------------------
# Fallback paths when model unavailable (lines 34-36, 58, 72-74, 91, 101-103)
# ---------------------------------------------------------------------------


class TestEmbedFallbacks:
    def test_embed_returns_zero_vector_when_model_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.embedder as emb_mod

        monkeypatch.setattr(emb_mod, "_model", None)
        monkeypatch.setattr(emb_mod, "_model_loaded", True)
        result = emb_mod.embed("some text")
        assert result.shape == (384,)
        assert np.all(result == 0.0)

    def test_embed_batch_returns_zero_vectors_when_model_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.embedder as emb_mod

        monkeypatch.setattr(emb_mod, "_model", None)
        monkeypatch.setattr(emb_mod, "_model_loaded", True)
        results = emb_mod.embed_batch(["text one", "text two"])
        assert len(results) == 2
        for arr in results:
            assert arr.shape == (384,)
            assert np.all(arr == 0.0)

    def test_get_model_returns_none_on_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        import core.embedder as emb_mod

        original_loaded = emb_mod._model_loaded
        original_model = emb_mod._model
        monkeypatch.setattr(emb_mod, "_model_loaded", False)
        monkeypatch.setattr(emb_mod, "_model", None)

        real_import = builtins.__import__

        def block_st(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_st)
        result = emb_mod._get_model()
        assert result is None
        monkeypatch.setattr(emb_mod, "_model_loaded", original_loaded)
        monkeypatch.setattr(emb_mod, "_model", original_model)

    def test_embed_handles_runtime_error_from_encode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.embedder as emb_mod

        class _FakeModel:
            def encode(self, *a: object, **kw: object) -> None:
                raise RuntimeError("CUDA OOM")

        monkeypatch.setattr(emb_mod, "_model", _FakeModel())
        monkeypatch.setattr(emb_mod, "_model_loaded", True)
        result = emb_mod.embed("test text")
        assert result.shape == (384,)
        assert np.all(result == 0.0)

    def test_embed_batch_handles_runtime_error_from_encode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.embedder as emb_mod

        class _FakeModel:
            def encode(self, *a: object, **kw: object) -> None:
                raise RuntimeError("CUDA OOM")

        monkeypatch.setattr(emb_mod, "_model", _FakeModel())
        monkeypatch.setattr(emb_mod, "_model_loaded", True)
        results = emb_mod.embed_batch(["text a", "text b"])
        assert len(results) == 2
        assert all(np.all(r == 0.0) for r in results)

    def test_embed_batch_squeezes_2d_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.embedder as emb_mod

        class _FakeModel:
            def encode(self, texts: list[str], **kw: object) -> np.ndarray:
                return np.ones((len(texts), 1, 384), dtype=np.float32)

        monkeypatch.setattr(emb_mod, "_model", _FakeModel())
        monkeypatch.setattr(emb_mod, "_model_loaded", True)
        results = emb_mod.embed_batch(["a"])
        assert results[0].shape == (384,)

    def test_embed_squeezes_2d_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.embedder as emb_mod

        class _FakeModel:
            def encode(self, text: str, **kw: object) -> np.ndarray:
                return np.ones((1, 384), dtype=np.float32)

        monkeypatch.setattr(emb_mod, "_model", _FakeModel())
        monkeypatch.setattr(emb_mod, "_model_loaded", True)
        result = emb_mod.embed("test")
        assert result.shape == (384,)
