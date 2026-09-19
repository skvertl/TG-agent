"""Unit tests for FastEmbed local embedding provider."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from adapters.embedding.fastembed_provider import FastEmbedProvider
from core.ports.embedding import BaseEmbeddingProvider


class TestFastEmbedProvider:
    def test_implements_interface(self):
        provider = FastEmbedProvider()
        assert isinstance(provider, BaseEmbeddingProvider)
        assert provider.dimension == 384

    @pytest.mark.asyncio
    async def test_embed_text_default_passage_prefix(self):
        provider = FastEmbedProvider()
        mock_model = MagicMock()
        mock_embedding = np.array([0.1] * 384)
        mock_model.embed.return_value = [mock_embedding]

        with patch.object(provider, "_get_model", return_value=mock_model):
            vec = await provider.embed_text("Hello document")
            assert len(vec) == 384
            mock_model.embed.assert_called_once_with(["passage: Hello document"])

    @pytest.mark.asyncio
    async def test_embed_query_prefix(self):
        provider = FastEmbedProvider()
        mock_model = MagicMock()
        mock_embedding = np.array([0.2] * 384)
        mock_model.embed.return_value = [mock_embedding]

        with patch.object(provider, "_get_model", return_value=mock_model):
            vec = await provider.embed_text("Where is the key?", is_query=True)
            assert len(vec) == 384
            mock_model.embed.assert_called_once_with(["query: Where is the key?"])

    @pytest.mark.asyncio
    async def test_no_double_prefixing(self):
        provider = FastEmbedProvider()
        mock_model = MagicMock()
        mock_embedding = np.array([0.1] * 384)
        mock_model.embed.return_value = [mock_embedding]

        with patch.object(provider, "_get_model", return_value=mock_model):
            await provider.embed_text("passage: Already prefixed")
            mock_model.embed.assert_called_once_with(["passage: Already prefixed"])

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        provider = FastEmbedProvider()
        mock_model = MagicMock()
        mock_model.embed.return_value = [
            np.array([0.1] * 384),
            np.array([0.2] * 384),
        ]

        with patch.object(provider, "_get_model", return_value=mock_model):
            vecs = await provider.embed_batch(["Doc 1", "Doc 2"])
            assert len(vecs) == 2
            assert len(vecs[0]) == 384
            assert len(vecs[1]) == 384
            mock_model.embed.assert_called_once_with(["passage: Doc 1", "passage: Doc 2"])

    @pytest.mark.asyncio
    async def test_embed_empty_batch(self):
        provider = FastEmbedProvider()
        vecs = await provider.embed_batch([])
        assert vecs == []
