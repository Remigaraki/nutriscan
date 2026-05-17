"""NutriScan — Streamlit UI  (Phase 6)

Run on the Pi:
    streamlit run nutriscan_app.py \
        --server.address 0.0.0.0 \
        --server.port 8501 \
        --server.headless true
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import requests
import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from cv_contract import validate_cv_output

# ---------------------------------------------------------------------------
# Configuration  (all overridable via environment variables)
# ---------------------------------------------------------------------------

_DB_PATH      = os.environ.get("NUTRISCAN_DB_PATH",  str(Path.home() / "nutriscan_db"))
_COLLECTION   = "health_guidelines"
_EMBED_MODEL  = "ncbi/MedCPT-Query-Encoder"
_OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")
_METRICS_FILE = os.environ.get("NUTRISCAN_METRICS", "/home/admin/nutriscan_metrics.json")
_TOP_K        = 3

# ---------------------------------------------------------------------------
# Quick-fill example payloads
# ---------------------------------------------------------------------------

_EXAMPLES: dict[str, dict] = {
    "🥣 Cereal": {
        "nutrition": {
            "calories": 250,
            "total_fat_g": 9.0,
            "saturated_fat_g": 3.5,
            "trans_fat_g": 0.0,
            "cholesterol_mg": 30,
            "sodium_mg": 470,
            "total_carbohydrate_g": 37.0,
            "dietary_fiber_g": 4.0,
            "total_sugars_g": 12.0,
            "added_sugars_g": 10.0,
            "protein_g": 5.0,
            "vitamin_d_mcg": 2.0,
            "calcium_mg": 260,
            "iron_mg": 8.0,
            "potassium_mg": 235,
        },
        "allergens": ["milk", "wheat", "soy"],
        "ingredients": [
            "whole grain oats", "sugar", "oat bran", "modified corn starch",
            "salt", "calcium carbonate", "niacinamide", "zinc and iron",
            "vitamin B6", "vitamin B2", "vitamin B1", "vitamin A",
            "folic acid", "vitamin B12", "vitamin D3",
        ],
    },
    "🍟 Chips": {
        "nutrition": {
            "calories": 150,
            "total_fat_g": 9.0,
            "saturated_fat_g": 1.5,
            "trans_fat_g": 0.0,
            "cholesterol_mg": 0,
            "sodium_mg": 230,
            "total_carbohydrate_g": 17.0,
            "dietary_fiber_g": 1.0,
            "total_sugars_g": 0.0,
            "protein_g": 2.0,
        },
        "allergens": [],
        "ingredients": [
            "potatoes", "vegetable oil", "salt", "dextrose",
            "sodium diacetate", "malic acid", "natural flavour",
        ],
    },
    "🥫 Canned Soup": {
        "nutrition": {
            "calories": 80,
            "total_fat_g": 1.5,
            "saturated_fat_g": 0.5,
            "trans_fat_g": 0.0,
            "cholesterol_mg": 5,
            "sodium_mg": 890,
            "total_carbohydrate_g": 12.0,
            "dietary_fiber_g": 2.0,
            "total_sugars_g": 3.0,
            "protein_g": 4.0,
        },
        "allergens": ["wheat", "soy", "milk"],
        "ingredients": [
            "chicken broth", "enriched egg noodles", "cooked chicken",
            "carrots", "celery", "modified food starch",
            "yeast extract", "soy protein isolate", "salt",
            "monosodium glutamate",
        ],
    },
}

# ---------------------------------------------------------------------------
# Cached resource initialisation (runs once per server process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBED_MODEL)


@st.cache_resource(show_spinner="Opening ChromaDB…")
def _load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_DB_PATH)
    return client.get_collection(_COLLECTION)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _embed_and_query(
    collection: chromadb.Collection,
    embedder: HuggingFaceEmbeddings,
    query: str,
) -> list[Document]:
    embedding = embedder.embed_query(query)
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=_TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    return [
        Document(
            page_content=text,
            metadata={**meta, "_distance": dist},
        )
        for text, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        )
    ]


def _retrieve(
    payload: dict,
    embedder: HuggingFaceEmbeddings,
    collection: chromadb.Collection,
) -> tuple[dict[str, list[Document]], float]:
    """Run three separate ChromaDB queries, one per CV field.

    Returns (hits_by_section, elapsed_seconds).
    """
    nutrition   = payload["nutrition"]
    allergens   = payload["allergens"]
    ingredients = payload["ingredients"]

    nutrition_q = (
        "Health guidelines for nutrients — "
        + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in nutrition.items())
        if nutrition else "Health guidelines for nutritional content"
    )
    allergens_q = (
        "Health risks and guidelines for allergens: " + ", ".join(allergens)
        if allergens else "Health guidelines for common food allergens"
    )
    ingredients_q = (
        "Health impact of food ingredients: " + ", ".join(ingredients[:10])
        if ingredients else "Health guidelines for food ingredients"
    )

    t0 = time.perf_counter()
    hits = {
        "Nutrition":   _embed_and_query(collection, embedder, nutrition_q),
        "Allergens":   _embed_and_query(collection, embedder, allergens_q),
        "Ingredients": _embed_and_query(collection, embedder, ingredients_q),
    }
    return hits, time.perf_counter() - t0


def _assemble_prompt(hits: dict[str, list[Document]]) -> str:
    sections = []
    for section, docs in hits.items():
        lines = [f"=== {section} Guidelines ==="]
        lines += [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1)]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> tuple[str, float, float]:
    """POST to Ollama and return (response_text, elapsed_s, tokens_per_sec).

    tokens_per_sec is derived from Ollama's own eval_count / eval_duration_ns.
    """
    t0 = time.perf_counter()
    resp = requests.post(
        f"{_OLLAMA_URL}/api/generate",
        json={"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()

    data        = resp.json()
    text        = data.get("response", "")
    eval_count  = data.get("eval_count", 0)
    eval_dur_ns = data.get("eval_duration", 0)
    tps         = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0

    return text, elapsed, tps


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

def _log(payload: dict, response: str, retrieval_s: float,
         llm_s: float, tps: float) -> None:
    record = {
        "ts":              datetime.now(timezone.utc).isoformat(),
        "allergens":       payload.get("allergens", []),
        "nutrition_keys":  list(payload.get("nutrition", {}).keys()),
        "ingredient_n":    len(payload.get("ingredients", [])),
        "retrieval_s":     round(retrieval_s, 3),
        "llm_s":           round(llm_s, 3),
        "tokens_per_sec":  round(tps, 1),
        "response_chars":  len(response),
    }
    try:
        with open(_METRICS_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # non-fatal: metrics path may not exist on dev machine


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NutriScan",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 NutriScan")
st.caption(
    f"Edge AI nutrition label analyser · Raspberry Pi 5 · "
    f"`{_OLLAMA_MODEL}`"
)

# Input mode selector
input_mode = st.radio(
    "input_mode",
    ["Manual JSON", "Scan Label"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

# --- Manual JSON panel ---
if input_mode == "Manual JSON":
    st.markdown("**Quick fill**")
    fill_cols = st.columns(len(_EXAMPLES))
    for col, (label, example) in zip(fill_cols, _EXAMPLES.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["_json_draft"] = json.dumps(example, indent=2)

    json_text: str = st.text_area(
        "CV Contract JSON",
        value=st.session_state.get("_json_draft", ""),
        height=320,
        placeholder=(
            '{\n'
            '  "nutrition": { "sodium_mg": 470, ... },\n'
            '  "allergens": ["milk", "wheat"],\n'
            '  "ingredients": ["whole grain oats", ...]\n'
            '}'
        ),
        label_visibility="collapsed",
    )

# --- Scan Label panel (stubbed) ---
else:
    # === CV INTEGRATION HOOK ===
    # Replace this section with the live camera capture + CV inference call
    # once the CV pipeline is ready (Phase 7).
    st.button("📷 Scan Label", disabled=True, use_container_width=False)
    st.info(
        "Camera scan is not yet available. "
        "This will trigger the CV pipeline in Phase 7.",
        icon="ℹ️",
    )
    json_text = ""

# --- Analyse button ---
st.divider()
analyse = st.button(
    "🔍 Analyse",
    type="primary",
    disabled=(input_mode == "Scan Label"),
)

if analyse:
    # ── 1. Parse ─────────────────────────────────────────────────────────────
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        st.error(f"**Invalid JSON** — {exc}")
        st.stop()

    # ── 2. Validate against CV contract ──────────────────────────────────────
    if not validate_cv_output(payload):
        st.error(
            "**Schema violation** — the JSON does not match the CV contract.  \n"
            "Required keys:  \n"
            "- `nutrition` — object whose values are numbers  \n"
            "- `allergens` — array of strings  \n"
            "- `ingredients` — array of strings"
        )
        st.stop()

    # ── 3. Load resources ────────────────────────────────────────────────────
    try:
        embedder   = _load_embedder()
        collection = _load_collection()
    except Exception as exc:
        st.error(f"**Failed to load resources:** {exc}")
        st.stop()

    # ── 4. Retrieve from ChromaDB ─────────────────────────────────────────────
    with st.spinner("Searching health guidelines…"):
        try:
            hits, retrieval_s = _retrieve(payload, embedder, collection)
        except Exception as exc:
            st.error(f"**Retrieval error:** {exc}")
            st.stop()

    # ── 5. Assemble prompt and call Ollama ────────────────────────────────────
    prompt = _assemble_prompt(hits)
    with st.spinner("Waiting for Ollama…"):
        try:
            llm_response, llm_s, tps = _call_ollama(prompt)
        except requests.exceptions.ConnectionError:
            st.error(
                f"**Ollama is not reachable** at `{_OLLAMA_URL}`.  \n"
                "On the Pi, run: `ollama serve` and confirm the model is pulled with  \n"
                f"`ollama pull {_OLLAMA_MODEL}`"
            )
            st.stop()
        except Exception as exc:
            st.error(f"**LLM error:** {exc}")
            st.stop()

    # ── 6. Log metrics ────────────────────────────────────────────────────────
    _log(payload, llm_response, retrieval_s, llm_s, tps)

    # ── 7. Display results ────────────────────────────────────────────────────
    st.subheader("Analysis")
    st.markdown(llm_response)

    # Citations panel
    with st.expander("📚 Citations", expanded=False):
        for section, docs in hits.items():
            st.markdown(f"**{section}**")
            for i, doc in enumerate(docs, 1):
                meta      = doc.metadata
                source    = meta.get("source", meta.get("filename", "unknown"))
                page      = meta.get("page", meta.get("chunk_id", "—"))
                distance  = float(meta.get("_distance", 1.0))
                relevance = max(0.0, min(1.0, 1.0 - distance))
                st.markdown(
                    f"&nbsp;&nbsp;`[{i}]` **{source}** · "
                    f"page {page} · "
                    f"relevance {relevance:.2f}"
                )
                st.caption(
                    doc.page_content[:400]
                    + ("…" if len(doc.page_content) > 400 else "")
                )

    # Performance panel
    with st.expander("⚡ Performance", expanded=False):
        p1, p2, p3 = st.columns(3)
        p1.metric("Retrieval", f"{retrieval_s * 1000:.0f} ms")
        p2.metric("LLM", f"{llm_s:.1f} s")
        p3.metric("Tokens / sec", f"{tps:.1f}" if tps > 0 else "n/a")
