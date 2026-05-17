"""NutriScan Router — routes CV payload fields to ChromaDB and assembles context for Ollama."""

import os
from typing import Any

import chromadb
import requests
from langchain_huggingface import HuggingFaceEmbeddings

from cv_contract import validate_cv_output

_DB_PATH = os.path.join(os.path.dirname(__file__), "nutriscan_db", "nutriscan_db")
_COLLECTION_NAME = "health_guidelines"
_EMBEDDING_MODEL = "ncbi/MedCPT-Query-Encoder"
_TOP_K = 3
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


# ---------------------------------------------------------------------------
# Query builders — each field gets a semantically appropriate query string
# ---------------------------------------------------------------------------

def _build_nutrition_query(nutrition: dict[str, Any]) -> str:
    if not nutrition:
        return "Health guidelines for nutritional content"
    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in nutrition.items()]
    return "Health guidelines for nutrients — " + ", ".join(parts)


def _build_allergens_query(allergens: list[str]) -> str:
    if not allergens:
        return "Health guidelines for common food allergens"
    return "Health risks and guidelines for allergens: " + ", ".join(allergens)


def _build_ingredients_query(ingredients: list[str]) -> str:
    if not ingredients:
        return "Health guidelines for food ingredients"
    preview = ingredients[:10]
    return "Health impact of food ingredients: " + ", ".join(preview)


# ---------------------------------------------------------------------------
# ChromaDB retrieval
# ---------------------------------------------------------------------------

def _embed_query(text: str, embedder: HuggingFaceEmbeddings) -> list[float]:
    return embedder.embed_query(text)


def _query_collection(
    collection: chromadb.Collection,
    embedding: list[float],
    top_k: int = _TOP_K,
) -> list[dict]:
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ---------------------------------------------------------------------------
# Ollama LLM call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = _OLLAMA_MODEL) -> str:
    """Send *prompt* to the local Ollama server and return the response text.

    Raises RuntimeError if Ollama is unreachable or returns an error.
    """
    try:
        response = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {_OLLAMA_URL}. "
            "Start Ollama with `ollama serve` before running the router."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route(cv_payload: dict) -> dict:
    """Route a validated CV payload through ChromaDB and Ollama.

    Returns a dict suitable for consumption by the Phase 6 Streamlit UI:

        {
            "nutrition_context":   list[{"text", "metadata", "distance"}],
            "allergens_context":   list[{"text", "metadata", "distance"}],
            "ingredients_context": list[{"text", "metadata", "distance"}],
            "assembled_prompt":    str,   # full context string sent to the LLM
            "llm_response":        str,   # Ollama's answer
        }

    Raises ValueError for invalid payloads.
    Raises RuntimeError if Ollama is unreachable.
    """
    if not validate_cv_output(cv_payload):
        raise ValueError("cv_payload does not conform to the CV contract.")

    embedder = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=_DB_PATH)
    collection = client.get_collection(_COLLECTION_NAME)

    # Build one query string per field and embed each independently
    nutrition_query = _build_nutrition_query(cv_payload["nutrition"])
    allergens_query = _build_allergens_query(cv_payload["allergens"])
    ingredients_query = _build_ingredients_query(cv_payload["ingredients"])

    nutrition_hits = _query_collection(collection, _embed_query(nutrition_query, embedder))
    allergens_hits = _query_collection(collection, _embed_query(allergens_query, embedder))
    ingredients_hits = _query_collection(collection, _embed_query(ingredients_query, embedder))

    def _fmt(hits: list[dict], section: str) -> str:
        lines = [f"=== {section} ==="]
        lines += [f"[{i}] {h['text']}" for i, h in enumerate(hits, 1)]
        return "\n".join(lines)

    assembled_prompt = "\n\n".join([
        _fmt(nutrition_hits, "Nutrition Guidelines"),
        _fmt(allergens_hits, "Allergen Guidelines"),
        _fmt(ingredients_hits, "Ingredient Guidelines"),
    ])

    llm_response = _call_ollama(assembled_prompt)

    return {
        "nutrition_context": nutrition_hits,
        "allergens_context": allergens_hits,
        "ingredients_context": ingredients_hits,
        "assembled_prompt": assembled_prompt,
        "llm_response": llm_response,
    }
