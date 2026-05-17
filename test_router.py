"""Tests for nutriscan_router.route().

Unit tests mock the embedder, ChromaDB, AND the Ollama call so no external
services are required.

The end-to-end test hits the real ChromaDB but still mocks Ollama (no server
running locally yet), and verifies the full output structure including
'llm_response'.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "cv_contract_example.json")
with open(_EXAMPLE_PATH) as _f:
    EXAMPLE_PAYLOAD = json.load(_f)

_MOCK_LLM_RESPONSE = "[mocked Ollama response]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_query_result(texts: list[str]) -> dict:
    return {
        "documents": [texts],
        "metadatas": [[{"source": f"doc{i}"} for i in range(len(texts))]],
        "distances": [[float(i) * 0.1 for i in range(len(texts))]],
    }


def _setup_mocks(MockEmbeddings, MockClient):
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.0] * 768
    MockEmbeddings.return_value = mock_emb

    mock_col = MagicMock()
    mock_col.query.return_value = _fake_query_result(["chunk A", "chunk B", "chunk C"])
    MockClient.return_value.get_collection.return_value = mock_col


# ---------------------------------------------------------------------------
# Unit tests — all external calls mocked
# ---------------------------------------------------------------------------

class TestRouteStructure(unittest.TestCase):

    def _patch_all(self):
        return (
            patch("nutriscan_router._call_ollama", return_value=_MOCK_LLM_RESPONSE),
            patch("nutriscan_router.chromadb.PersistentClient"),
            patch("nutriscan_router.HuggingFaceEmbeddings"),
        )

    def test_output_keys_present(self):
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)

        for key in ("nutrition_context", "allergens_context",
                    "ingredients_context", "assembled_prompt", "llm_response"):
            self.assertIn(key, result)

    def test_each_context_has_three_results(self):
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)

        self.assertEqual(len(result["nutrition_context"]), 3)
        self.assertEqual(len(result["allergens_context"]), 3)
        self.assertEqual(len(result["ingredients_context"]), 3)

    def test_context_items_have_expected_fields(self):
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)

        for section in ("nutrition_context", "allergens_context", "ingredients_context"):
            for item in result[section]:
                self.assertIn("text", item)
                self.assertIn("metadata", item)
                self.assertIn("distance", item)

    def test_assembled_prompt_is_nonempty_string(self):
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)

        self.assertIsInstance(result["assembled_prompt"], str)
        self.assertGreater(len(result["assembled_prompt"]), 0)

    def test_assembled_prompt_contains_section_headers(self):
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)

        prompt = result["assembled_prompt"]
        self.assertIn("Nutrition Guidelines", prompt)
        self.assertIn("Allergen Guidelines", prompt)
        self.assertIn("Ingredient Guidelines", prompt)

    def test_llm_response_is_mocked_string(self):
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)

        self.assertEqual(result["llm_response"], _MOCK_LLM_RESPONSE)

    def test_assembled_prompt_is_passed_to_ollama(self):
        """_call_ollama must receive the assembled_prompt as its first argument."""
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama as mock_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)
            mock_ollama.assert_called_once_with(result["assembled_prompt"])

    def test_invalid_payload_raises_value_error(self):
        with patch("nutriscan_router._call_ollama", return_value=_MOCK_LLM_RESPONSE):
            from nutriscan_router import route
            with self.assertRaises(ValueError):
                route({"nutrition": {}, "allergens": []})  # missing ingredients

    def test_empty_allergens_and_ingredients(self):
        payload = {"nutrition": {"sodium_mg": 500}, "allergens": [], "ingredients": []}
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            from nutriscan_router import route
            result = route(payload)
        self.assertIn("assembled_prompt", result)
        self.assertIn("llm_response", result)

    def test_collection_queried_three_times(self):
        """One query per field — nutrition, allergens, ingredients."""
        p_ollama, p_client, p_emb = self._patch_all()
        with p_ollama, p_client as MockClient, p_emb as MockEmbeddings:
            _setup_mocks(MockEmbeddings, MockClient)
            mock_col = MockClient.return_value.get_collection.return_value
            from nutriscan_router import route
            route(EXAMPLE_PAYLOAD)
            self.assertEqual(mock_col.query.call_count, 3)


# ---------------------------------------------------------------------------
# End-to-end test — real ChromaDB, mocked Ollama
# ---------------------------------------------------------------------------

class TestRouteEndToEnd(unittest.TestCase):
    """Hits the real ChromaDB; Ollama is mocked (not yet running locally)."""

    def test_e2e_route_output_structure(self):
        with patch("nutriscan_router._call_ollama", return_value=_MOCK_LLM_RESPONSE):
            from nutriscan_router import route
            result = route(EXAMPLE_PAYLOAD)

        # All five output keys must be present
        for key in ("nutrition_context", "allergens_context",
                    "ingredients_context", "assembled_prompt", "llm_response"):
            self.assertIn(key, result, f"Missing key: {key}")

        # Each context section returns 1–3 real chunks from the DB
        for section in ("nutrition_context", "allergens_context", "ingredients_context"):
            hits = result[section]
            self.assertIsInstance(hits, list)
            self.assertGreater(len(hits), 0)
            self.assertLessEqual(len(hits), 3)
            for hit in hits:
                self.assertIn("text", hit)
                self.assertIsInstance(hit["text"], str)
                self.assertGreater(len(hit["text"]), 0)
                self.assertIn("distance", hit)
                self.assertIsInstance(hit["distance"], float)

        # Assembled prompt is non-trivial
        self.assertIsInstance(result["assembled_prompt"], str)
        self.assertGreater(len(result["assembled_prompt"]), 50)
        self.assertIn("Nutrition Guidelines", result["assembled_prompt"])

        # LLM response is the mocked value
        self.assertEqual(result["llm_response"], _MOCK_LLM_RESPONSE)


if __name__ == "__main__":
    unittest.main()
