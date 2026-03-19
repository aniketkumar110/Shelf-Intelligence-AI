"""
main.py

Streamlit app for:
- Upload/select image from Azure Blob
- Vector search images via Azure AI Search (text -> embedding -> vector query)
- GPT-4o validation of vector-search results (only matching images shown)
- Run vision pipeline (row detect + crop + product detect)
- Run agentic LLM chain on row images
- Display final answer + reasoning + pipeline visuals

UI: polished gallery with checkbox overlay, SAS-URL lazy loading, no full-reload.
"""

from __future__ import annotations

import os
import re
import json
import time
import uuid
import shutil
import logging
import base64
import urllib.parse 
import asyncio
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from io import BytesIO
from semantic import index_all_images_from_blob 
from semantic import index_all_images_from_blob, index_single_blob_sync

import streamlit as st
import aiohttp

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from detect_crop import detect_and_crop_rows, annotate_rows_with_yolo
from llm_processing import create_llm_orchestrator
from blob_pipeline_trigger import (
    run_pipeline_and_store,
    fetch_processed_images,
    is_already_processed,
    get_pipeline_metadata,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# GALLERY CSS  (injected once at startup)
# ─────────────────────────────────────────────────────────────────────
GALLERY_CSS = """
<style>
/* ── Import font ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

/* ── Root variables ───────────────────────────────────── */
:root {
    --accent: #4f8ef7;
    --accent-glow: rgba(79, 142, 247, 0.35);
    --card-bg: #111827;
    --card-border: #1f2937;
    --card-hover: #1e2d45;
    --selected-border: #4f8ef7;
    --radius: 10px;
    --img-height: 108px;
}

/* ── Global font ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Gallery wrapper: scrollable right panel ──────────── */
.gallery-scroll {
    max-height: 72vh;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: #374151 transparent;
}
.gallery-scroll::-webkit-scrollbar { width: 4px; }
.gallery-scroll::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }

/* ── Image card ───────────────────────────────────────── */
.g-card {
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--card-bg);
    border: 2px solid var(--card-border);
    transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.12s ease;
    cursor: pointer;
    margin-bottom: 6px;
    height: var(--img-height);
}
.g-card:hover {
    border-color: #374151;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transform: translateY(-1px);
}
.g-card.g-selected {
    border-color: var(--selected-border);
    box-shadow: 0 0 0 3px var(--accent-glow), 0 4px 20px rgba(0,0,0,0.4);
}
.g-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    border-radius: calc(var(--radius) - 2px);
}

/* ── Checkbox overlay ─────────────────────────────────── */
.g-card.g-selected .g-check {
    background: var(--accent);
    border-color: var(--accent);
}

/* ── Caption / blob name ──────────────────────────────── */
.g-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 3px 6px;
    font-size: 9px;
    color: rgba(255,255,255,0.75);
    background: linear-gradient(transparent, rgba(0,0,0,0.65));
    border-radius: 0 0 calc(var(--radius) - 2px) calc(var(--radius) - 2px);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Pagination controls ──────────────────────────────── */
.pager {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    font-size: 12px;
    color: #9ca3af;
}

/* ── Search input polish ──────────────────────────────── */
.stTextInput input {
    border-radius: 8px !important;
    border-color: #374151 !important;
    background: #111827 !important;
    color: #f9fafb !important;
    font-size: 13px !important;
}

/* ── Tab styling ──────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Selected file badge ──────────────────────────────── */
.sel-badge {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(79,142,247,0.15);
    border: 1px solid rgba(79,142,247,0.4);
    border-radius: 20px;
    font-size: 11px;
    color: #93c5fd;
    margin-bottom: 8px;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Streamlit column gap override ───────────────────── */
[data-testid="column"] {
    padding: 0 3px !important;
}

/* ── GPT validation badge ─────────────────────────────── */
.val-badge-pass {
    display: inline-block;
    padding: 2px 8px;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 12px;
    font-size: 10px;
    color: #86efac;
    margin-top: 3px;
    cursor: help;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    return v or ""


def load_app_settings() -> Dict[str, str]:
    settings = {
        "AZURE_BLOB_CONN": _env("AZURE_BLOB_CONN"),
        "AZURE_BLOB_CONTAINER": _env("AZURE_BLOB_CONTAINER", "vision-agent"),
        "UPLOAD_DIR": _env("UPLOAD_DIR", "Input"),
        "AZURE_SEARCH_ENDPOINT1": _env("AZURE_SEARCH_ENDPOINT1"),
        "AZURE_SEARCH_ADMIN_KEY1": _env("AZURE_SEARCH_ADMIN_KEY1"),
        "AZURE_SEARCH_INDEX_NAME2": _env("AZURE_SEARCH_INDEX_NAME2", "image_embeddings_index"),
        "AZURE_AI_VISION_API_KEY": _env("AZURE_AI_VISION_API_KEY"),
        "AZURE_AI_VISION_REGION": _env("AZURE_AI_VISION_REGION", "eastus"),
        "AZURE_AI_VISION_ENDPOINT": _env("AZURE_AI_VISION_ENDPOINT"),
        "AZURE_OPENAI_ENDPOINT":    _env("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY":     _env("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_API_VERSION": _env("AZURE_OPENAI_API_VERSION"),
        "AZURE_OPENAI_DEPLOYMENT":  _env("AZURE_OPENAI_DEPLOYMENT"),
    }
        

    return settings


SAFE_BLOB_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-./]+")


def sanitize_blob_name(name: str) -> str:
    name = name.replace("\\", "/")
    name = SAFE_BLOB_NAME_RE.sub("_", name)
    return name.lstrip("/").replace("../", "_").replace("..", "_")


def _b64key(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii")

def _sync_env_for_semantic(s: Dict[str, str]) -> None:
    """Pushes app settings into os.environ so semantic.py can read them."""
    os.environ.setdefault("AZURE_SEARCH_ENDPOINT1",    s["AZURE_SEARCH_ENDPOINT1"])
    os.environ.setdefault("AZURE_SEARCH_ADMIN_KEY1",   s["AZURE_SEARCH_ADMIN_KEY1"])
    os.environ.setdefault("AZURE_SEARCH_INDEX_NAME2",  s["AZURE_SEARCH_INDEX_NAME2"])
    os.environ.setdefault("AZURE_AI_VISION_API_KEY",   s["AZURE_AI_VISION_API_KEY"])
    os.environ.setdefault("AZURE_AI_VISION_REGION",    s["AZURE_AI_VISION_REGION"])
    os.environ.setdefault("AZURE_AI_VISION_ENDPOINT",  s["AZURE_AI_VISION_ENDPOINT"])
    os.environ.setdefault("AZURE_STORAGE_CONNECTION_STRING", s["AZURE_BLOB_CONN"])
    os.environ.setdefault("AZURE_STORAGE_CONTAINER",   s["AZURE_BLOB_CONTAINER"])

# ─────────────────────────────────────────────────────────────────────
# CACHED SINGLETONS
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_blob_service() -> BlobServiceClient:
    s = load_app_settings()
    return BlobServiceClient.from_connection_string(s["AZURE_BLOB_CONN"])


@st.cache_resource
def get_blob_container():
    s = load_app_settings()
    return get_blob_service().get_container_client(s["AZURE_BLOB_CONTAINER"])


@st.cache_resource
def get_search_client() -> SearchClient:
    s = load_app_settings()
    return SearchClient(
        endpoint=s["AZURE_SEARCH_ENDPOINT1"],
        index_name=s["AZURE_SEARCH_INDEX_NAME2"],
        credential=AzureKeyCredential(s["AZURE_SEARCH_ADMIN_KEY1"]),
    )


@st.cache_resource
def get_llm():
    return create_llm_orchestrator()


@st.cache_resource
def get_openai_client() -> AzureOpenAI:
    """Cached AzureOpenAI client used for GPT-4o image validation."""
    s = load_app_settings()
    if not s["AZURE_OPENAI_ENDPOINT"] or not s["AZURE_OPENAI_API_KEY"]:
        raise RuntimeError(
            "Azure OpenAI credentials missing. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
        )
    return AzureOpenAI(
        azure_endpoint=s["AZURE_OPENAI_ENDPOINT"],
        api_key=s["AZURE_OPENAI_API_KEY"],
        api_version=s["AZURE_OPENAI_API_VERSION"],
    )


# ─────────────────────────────────────────────────────────────────────
# BLOB HELPERS
# ─────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=300, show_spinner=False)
# def list_blobs_images_cached(conn_str: str, container: str) -> List[str]:
#     """Cached blob listing — avoids repeated Azure API calls on every render."""
#     bsc = BlobServiceClient.from_connection_string(conn_str)
#     cc = bsc.get_container_client(container)
#     return [
#         b.name for b in cc.list_blobs()
#         if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]
@st.cache_data(ttl=300, show_spinner=False)
def list_blobs_images_cached(conn_str: str, container: str) -> List[str]:
    _PIPELINE_DIRS = {"cropped_images", "products_detected", "original"}
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)
    results = []
    for b in cc.list_blobs():
        name: str = b.name
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = name.split("/")
        if len(parts) >= 2 and parts[-2] in _PIPELINE_DIRS:
            continue
        results.append(name)
    return results


@st.cache_data(show_spinner=False)
def download_blob_bytes_cached(conn_str: str, container: str, blob_name: str) -> bytes:
    """Download full blob bytes, cached per blob_name."""
    bsc = BlobServiceClient.from_connection_string(conn_str)
    bc = bsc.get_blob_client(container=container, blob=blob_name)
    return bc.download_blob().readall()


@st.cache_data(ttl=1500, show_spinner=False)
def get_blob_sas_url_cached(conn_str: str, container: str, blob_name: str) -> str:
    """25-min cached SAS URL — browser fetches image direct from Azure."""
    bsc = BlobServiceClient.from_connection_string(conn_str)
    parts = dict(p.split("=", 1) for p in conn_str.split(";") if "=" in p)
    sas = generate_blob_sas(
        account_name=bsc.account_name,
        account_key=parts.get("AccountKey", ""),
        container_name=container,
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.datetime.utcnow() + datetime.timedelta(minutes=30),
    )
    return (
        f"https://{bsc.account_name}.blob.core.windows.net"
        f"/{container}/{urllib.parse.quote(blob_name)}?{sas}"
    )


def upload_to_blob(conn_str: str, container: str, blob_name: str, data: bytes) -> None:
    """Upload raw bytes to Azure Blob; overwrites if exists."""
    bsc = BlobServiceClient.from_connection_string(conn_str)
    bc = bsc.get_blob_client(container=container, blob=blob_name)
    bc.upload_blob(data, overwrite=True)
    logger.info(f"Uploaded '{blob_name}' → container '{container}'")


def invalidate_blob_list_cache(conn_str: str, container: str) -> None:
    """Force-clear the cached blob listing (triggers fresh fetch on next render)."""
    list_blobs_images_cached.clear()


# ─────────────────────────────────────────────────────────────────────
# VECTOR SEARCH  (Step 1 — retrieval only, no validation)
# ─────────────────────────────────────────────────────────────────────
def _normalize_region(r: str) -> str:
    return r.strip().lower().replace(" ", "") if r else r


def _vision_url(region: str, endpoint: str, url_path: str) -> str:
    if endpoint and endpoint.strip():
        parsed = urllib.parse.urlparse(endpoint.strip())
        if not parsed.netloc:
            raise ValueError(f"Invalid AZURE_AI_VISION_ENDPOINT: {endpoint}")
        return f"{endpoint.rstrip('/')}{url_path}"
    return f"https://{region}.api.cognitive.microsoft.com{url_path}"


async def _embed_text_async(prompt: str, api_key: str, region: str, endpoint: str) -> List[float]:
    headers = {"Ocp-Apim-Subscription-Key": api_key, "Content-Type": "application/json"}
    params = urllib.parse.urlencode({"model-version": "2023-04-15"})
    url = _vision_url(region, endpoint,
        f"/computervision/retrieval:vectorizeText?api-version=2024-02-01&{params}")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json={"text": prompt}, timeout=60) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Vision vectorizeText {resp.status}: {await resp.text()}")
            data = await resp.json()
    vec = data.get("vector")
    if not isinstance(vec, list):
        raise RuntimeError("No 'vector' in Vision response.")
    return vec


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    except RuntimeError:
        return asyncio.run(coro)


@st.cache_data(ttl=3600, show_spinner=False)
def embed_query_text(prompt: str) -> List[float]:
    s = load_app_settings()
    # Add this block right after: s = load_app_settings()
    if not st.session_state.get("blob_cache_cleared"):
        list_blobs_images_cached.clear()
        st.session_state["blob_cache_cleared"] = True
    region = _normalize_region(s["AZURE_AI_VISION_REGION"])
    return _run_coro(_embed_text_async(
        prompt, s["AZURE_AI_VISION_API_KEY"], region, s["AZURE_AI_VISION_ENDPOINT"]
    ))


def vector_search_blobs(prompt: str, top_k: int) -> List[Dict]:
    """
    Step 1 — Pure vector search.
    Returns raw candidate list; no GPT validation here.
    """
    sc = get_search_client()
    query_vec = embed_query_text(prompt)
    vq = VectorizedQuery(vector=query_vec, k_nearest_neighbors=top_k, fields="image_vector")
    results = sc.search(
        search_text="",
        vector_queries=[vq],
        select=["container", "blob_name"],
        top=top_k,
    )
    return [
        {
            "score": r.get("@search.score"),
            "container": r.get("container"),
            "blob_name": r.get("blob_name"),
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────────
# GPT-4o IMAGE VALIDATION  (Step 2 — filter by relevance)
# ─────────────────────────────────────────────────────────────────────
def _image_bytes_to_base64(image_bytes: bytes) -> Tuple[str, str]:
    """Convert raw bytes → (base64_string, mime_type)."""
    img = Image.open(BytesIO(image_bytes))
    fmt = (img.format or "JPEG").upper()
    mime = {
        "JPEG": "image/jpeg", "JPG": "image/jpeg",
        "PNG": "image/png", "WEBP": "image/webp", "BMP": "image/bmp",
    }.get(fmt, "image/jpeg")
    buf = BytesIO()
    img.save(buf, format=fmt if fmt != "JPG" else "JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), mime


def validate_single_image_with_gpt(
    image_bytes: bytes,
    user_query: str,
    blob_name: str,
) -> Tuple[bool, str]:
    """
    Ask GPT-4o whether this image visually matches the user query.
    Uses 'detail: low' to keep latency and cost minimal.
    Returns: (is_relevant: bool, reason: str)
    """
    s = load_app_settings()
    client = get_openai_client()
    deployment = s["AZURE_OPENAI_DEPLOYMENT"]

    b64, mime = _image_bytes_to_base64(image_bytes)

    system_prompt = (
        "You are a strict image relevance checker for a retail planogram system. "
        "Your only job is to decide if a shelf/store image visually matches a user's search query. "
        "Reply ONLY with a JSON object in the form: "
        "{\"relevant\": true, \"reason\": \"one sentence\"} "
        "or {\"relevant\": false, \"reason\": \"one sentence\"}. "
        "No markdown, no extra text."
    )

    user_prompt = (
        f"Search query: \"{user_query}\"\n\n"
        "Does this shelf image match the search query? "
        "Look for the products, brands, shelf type, or visual context described. "
        "Be strict — only approve images that clearly relate to the query."
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            max_completion_tokens=120,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        return bool(parsed.get("relevant", False)), str(parsed.get("reason", ""))

    except Exception as e:
        logger.warning(f"validation error for '{blob_name}': {e}")
        return True, f"Validation error (included by default): {e}"


def validate_search_results_with_gpt(
    candidates: List[Dict],
    user_query: str,
    conn_str: str,
    progress_placeholder,
) -> List[Dict]:
    """
    Step 2 — Parallel GPT-4o validation using ThreadPoolExecutor.
    All candidates are validated concurrently; only relevant ones are returned.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(candidates)
    validated: List[Dict] = []
    completed_count = 0
    lock = __import__("threading").Lock()

    progress_placeholder.caption(f"🤖 validating {total} images")

    def validate_one(item: Dict):
        blob_name = item.get("blob_name", "")
        container = item.get("container", "")
        if not blob_name:
            return None
        try:
            image_bytes = download_blob_bytes_cached(conn_str, container, blob_name)
            is_relevant, reason = validate_single_image_with_gpt(
                image_bytes, user_query, blob_name
            )
        except Exception as e:
            logger.warning(f"Skipping '{blob_name}': {e}")
            is_relevant, reason = False, str(e)

        if is_relevant:
            return {**item, "gpt_validated": True, "gpt_reason": reason}
        return None

    with ThreadPoolExecutor(max_workers=min(total, 10)) as executor:
        futures = {executor.submit(validate_one, item): item for item in candidates}
        for future in as_completed(futures):
            with lock:
                completed_count += 1
                progress_placeholder.caption(
                    f"🤖 validating… {completed_count} / {total} done"
                )
            result = future.result()
            if result is not None:
                with lock:
                    validated.append(result)

    progress_placeholder.empty()
    return validated


# ─────────────────────────────────────────────────────────────────────
# TEMP DIR HELPERS
# ─────────────────────────────────────────────────────────────────────
def ensure_session_run_dirs() -> Dict[str, Path]:
    if "run_id" not in st.session_state or not st.session_state["run_id"]:
        st.session_state["run_id"] = str(uuid.uuid4())
    run_id = st.session_state["run_id"]
    root = Path("temp_run") / run_id
    dirs = {
        "ROOT": root,
        "ROW_DIR": root / "cropped_rows",
        "ANNOTATED_DIR": root / "annotated_rows",
        "SUMMARY_OUT": root / "shelf_summary.jpg",
    }
    for k, p in dirs.items():
        if k.endswith("_DIR") or k == "ROOT":
            p.mkdir(parents=True, exist_ok=True)
    return dirs


def reset_run_dirs():
    if "run_id" in st.session_state and st.session_state["run_id"]:
        root = Path("temp_run") / st.session_state["run_id"]
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    st.session_state["run_id"] = str(uuid.uuid4())
    ensure_session_run_dirs()


# ─────────────────────────────────────────────────────────────────────
# VISION + LLM
# ─────────────────────────────────────────────────────────────────────
# def process_image_once(
#     image_path: str,
#     dirs: Dict[str, Path],
#     blob_name: Optional[str] = None,
#     conn_str: Optional[str] = None,
#     container: Optional[str] = None,
# ):
#     """
#     Returns pipeline outputs.

#     Priority order:
#       1. If blob_name is given and pre-processed outputs exist in blob →
#          download and use them (no Roboflow call).
#       2. Otherwise → run Roboflow locally and return results (legacy fallback,
#          should only happen if the upload-time trigger failed).
#     """
#     timings: Dict[str, float] = {}

#     # ── Try fetching pre-processed results from blob ──────────────────
#     if blob_name and conn_str and container:
#         if is_already_processed(conn_str, container, blob_name):
#             t0 = time.time()
#             cropped_paths, annotated_paths, row_counts = fetch_processed_images(
#                 conn_str=conn_str,
#                 container=container,
#                 blob_name=blob_name,
#                 local_dir=str(dirs["ROOT"]),
#                 row_dir=str(dirs["ROW_DIR"]),
#                 annotated_dir=str(dirs["ANNOTATED_DIR"]),
#             )
#             timings["Fetch from blob (pre-processed)"] = time.time() - t0

#             if annotated_paths:
#                 timings["Total vision pipeline"] = timings["Fetch from blob (pre-processed)"]
#                 return annotated_paths, row_counts, None, timings
def process_image_once(
    image_path: str,
    dirs: Dict[str, Path],
    blob_name: Optional[str] = None,
    conn_str: Optional[str] = None,
    container: Optional[str] = None,
):
    timings: Dict[str, float] = {}

    # ── Fast path: files already downloaded in this session ──────────
    existing_annotated = sorted(dirs["ANNOTATED_DIR"].glob("*.png"))
    existing_cropped = sorted(dirs["ROW_DIR"].glob("*.png"))
    if existing_annotated and existing_cropped:
        logger.info("Using already-downloaded local files (session cache).")
        timings["Total vision pipeline"] = 0.0
        return (
            [str(p) for p in existing_annotated],
            [],   # row_counts not needed again
            None,
            timings,
        )

    # ── Try fetching pre-processed results from blob ──────────────────
    if blob_name and conn_str and container:
        if is_already_processed(conn_str, container, blob_name):
            t0 = time.time()
            cropped_paths, annotated_paths, row_counts = fetch_processed_images(
                conn_str=conn_str,
                container=container,
                blob_name=blob_name,
                local_dir=str(dirs["ROOT"]),
                row_dir=str(dirs["ROW_DIR"]),
                annotated_dir=str(dirs["ANNOTATED_DIR"]),
            )
            timings["Fetch from blob (pre-processed)"] = time.time() - t0

            if annotated_paths:
                timings["Total vision pipeline"] = timings["Fetch from blob (pre-processed)"]
                return annotated_paths, row_counts, None, timings

    # ── Fallback: run Roboflow locally ────────────────────────────────
    # ... rest unchanged
    # ── Fallback: run Roboflow locally ────────────────────────────────
    logger.warning(
        "Pre-processed outputs not found for '%s'. Running Roboflow locally (slow path).",
        blob_name,
    )
    t0 = time.time()
    row_paths, error = detect_and_crop_rows(
        image_path=image_path,
        summary_out=str(dirs["SUMMARY_OUT"]),
        crop_dir=str(dirs["ROW_DIR"]),
    )
    timings["Row detection + cropping"] = time.time() - t0
    if error:
        return [], [], error, timings
    if not row_paths:
        return [], [], "No cropped rows produced.", timings
    t1 = time.time()
    annotated_paths, row_counts = annotate_rows_with_yolo(
        row_paths=row_paths, output_dir=str(dirs["ANNOTATED_DIR"])
    )
    timings["Product detection"] = time.time() - t1
    timings["Total vision pipeline"] = timings["Row detection + cropping"] + timings["Product detection"]
    return annotated_paths, row_counts, None, timings


# ─────────────────────────────────────────────────────────────────────
# GALLERY RENDERER
# ─────────────────────────────────────────────────────────────────────
def render_gallery(
    items: List[Dict],
    current_file_path: str,
    upload_dir: Path,
    s: Dict,
    key_prefix: str = "gal",
    show_gpt_reason: bool = False,
) -> bool:
    """
    4-column gallery with checkbox selection.
    Returns True if an image was selected (caller must st.rerun()).
    """
    COLS = 4
    rows = [items[i: i + COLS] for i in range(0, len(items), COLS)]

    for row in rows:
        cols = st.columns(COLS, gap="small")

        for col, item in zip(cols, row):
            blob_name = item["blob_name"]
            cont = item.get("container") or s["AZURE_BLOB_CONTAINER"]
            sas_url = item["sas_url"]
            gpt_reason = item.get("gpt_reason", "")

            safe_name = sanitize_blob_name(blob_name)
            img_stem = Path(safe_name).stem
            local_path = str(upload_dir / img_stem / safe_name)
            is_selected = (current_file_path == local_path)
            checkbox_key = f"{key_prefix}_{_b64key(cont + '::' + blob_name)}"

            st.session_state.setdefault("selected_blob_key", None)

            with col:
                st.markdown(
                    f"""
                    <div style="height:108px; overflow:hidden; border-radius:8px; margin-bottom:4px;">
                        <img src="{sas_url}" style="width:100%; height:108px; object-fit:cover; border-radius:8px; display:block;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                checked = st.checkbox("", key=checkbox_key, value=is_selected, label_visibility="collapsed")
                if checked and not is_selected:
                    with st.spinner(""):
                        try:
                            img_bytes = download_blob_bytes_cached(
                                s["AZURE_BLOB_CONN"], cont, blob_name
                            )
                        except Exception as e:
                            st.error(f"Download failed: {e}")
                            return False

                    lp = upload_dir / img_stem / safe_name
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    lp.write_bytes(img_bytes)
                    st.session_state.file_path = str(lp)
                    st.session_state.image_processed = False

                    # Kick off pipeline for this blob if not already processed
                    if not is_already_processed(s["AZURE_BLOB_CONN"], cont, blob_name):
                        with st.spinner("🤖 Running vision pipeline & caching results…"):
                            import threading as _threading
                            _err_holder: Dict = {}

                            def _bg_pipeline():
                                try:
                                    success, err = run_pipeline_and_store(
                                        conn_str=s["AZURE_BLOB_CONN"],
                                        container=cont,
                                        blob_name=blob_name,
                                        local_image_path=str(lp),
                                    )
                                    if not success:
                                        _err_holder["error"] = err
                                except Exception as exc:
                                    _err_holder["error"] = str(exc)

                            _t = _threading.Thread(target=_bg_pipeline, daemon=False)
                            _t.start()
                            _t.join()

                            if "error" in _err_holder:
                                st.warning(f"⚠️ Vision pipeline failed: {_err_holder['error']}")

                    return True   # caller calls st.rerun()

    return False


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="Planogram Vision Agent", layout="wide")
    st.markdown(GALLERY_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-top: -70px;'>
        <h2 style='font-size: 40px; font-family: Courier New, monospace;'>
            <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" width="70" height="60">
            <span style='background: linear-gradient(45deg, #ed4965, #c05aaf); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Planogram Vision Agent
            </span>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    s = load_app_settings()
    upload_dir = Path(s["UPLOAD_DIR"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    

    # ── Session state init ──────────────────────────────────────────
    for key in [
        "file_path", "image_processed", "annotated_paths", "row_counts",
        "final_answer", "final_reasoning", "latency_stats",
        "vector_results", "vector_query_text", "vector_top_k",
        "vector_raw_count",
    ]:
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("browse_page", 0)
    st.session_state.setdefault("active_tab", 0)
    st.session_state.setdefault("selected_blob_key", None)

    dirs = ensure_session_run_dirs()
    llm = get_llm()

    st.markdown("## 📤 Upload or Select Image")
    mode = st.radio("Choose Input Mode:", ["📤 Upload File", "📁 Select From Blob"], horizontal=True)

    blob_files = list_blobs_images_cached(s["AZURE_BLOB_CONN"], s["AZURE_BLOB_CONTAINER"])

    if mode == "📁 Select From Blob":
        left_col, right_col = st.columns([3, 1])
    else:
        left_col = st.container()
        right_col = None

    # ═══════════════════════════════════════════════════════════════
    # LEFT PANEL
    # ═══════════════════════════════════════════════════════════════
    with left_col:
        file_to_use: Optional[str] = None

        # ── UPLOAD FILE MODE ────────────────────────────────────────
        if mode == "📤 Upload File":
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                blob_name = sanitize_blob_name(uploaded_file.name)
                file_bytes = uploaded_file.read()

                # 1. Save locally inside a per-image subfolder: UPLOAD_DIR/<stem>/<filename>
                img_stem = Path(blob_name).stem
                dest = upload_dir / img_stem / blob_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(file_bytes)
                file_to_use = str(dest)

                # 2. Upload to Azure Blob Storage (overwrite=True is safe and idempotent)
                try:
                    with st.spinner("☁️ Uploading to Azure Blob Storage…"):
                        upload_to_blob(
                            conn_str=s["AZURE_BLOB_CONN"],
                            container=s["AZURE_BLOB_CONTAINER"],
                            blob_name=blob_name,
                            data=file_bytes,
                        )

                    # 3. Bust the blob-listing cache so Browse tab reflects the new image
                    invalidate_blob_list_cache(s["AZURE_BLOB_CONN"], s["AZURE_BLOB_CONTAINER"])

                    # 4. Run Roboflow pipeline immediately and persist outputs to blob
                    #    so subsequent queries use cached results instead of re-running.
                    if not is_already_processed(s["AZURE_BLOB_CONN"], s["AZURE_BLOB_CONTAINER"], blob_name):
                        with st.spinner("🤖 Running.."):
                            import threading
                            pipeline_error_holder: Dict = {}

                            def _run_pipeline():
                                try:
                                    success, err = run_pipeline_and_store(
                                        conn_str=s["AZURE_BLOB_CONN"],
                                        container=s["AZURE_BLOB_CONTAINER"],
                                        blob_name=blob_name,
                                        local_image_path=str(dest),
                                    )
                                    if not success:
                                        pipeline_error_holder["error"] = err
                                except Exception as exc:
                                    pipeline_error_holder["error"] = str(exc)

                            t = threading.Thread(target=_run_pipeline, daemon=False)
                            t.start()
                            t.join()  # wait so results are ready before user queries

                            if "error" in pipeline_error_holder:
                                st.warning(
                                    f"⚠️ Vision pipeline partially failed: {pipeline_error_holder['error']}. "
                                    "Will re-run at query time."
                                )

                except Exception as e:
                    st.warning(f"⚠️ Saved locally but Blob upload failed: {e}")

                # else:
                #     st.success("✅ Vision pipeline complete — results cached in blob.")

                # ADD THIS RIGHT AFTER:
                with st.spinner("🔍 Indexing..."):
                    try:
                        _sync_env_for_semantic(s)  
                        index_single_blob_sync(
                            blob_name=blob_name,
                            container=s["AZURE_BLOB_CONTAINER"],
                        )
                        #st.success("✅ Image indexed for semantic search.")
                    except Exception as e:
                        st.warning(f"⚠️ Indexing failed: {e}")

        # ── BLOB SELECT MODE — file set via gallery checkbox ────────
        elif mode == "📁 Select From Blob":
            file_to_use = st.session_state.file_path

        # ── Sync file_to_use → session state ───────────────────────
        if file_to_use:
            if st.session_state.file_path != file_to_use:
                reset_run_dirs()
                st.session_state.image_processed = False
                st.session_state.final_answer = ""
                st.session_state.final_reasoning = ""
                st.session_state.latency_stats = {}
            st.session_state.file_path = file_to_use

            with st.expander("📂 Preview Image", expanded=True):
                st.image(file_to_use, width=350)

        # ── Question input ──────────────────────────────────────────
        st.markdown("### 💬 Ask Your Question")
        if mode == "📁 Select From Blob":
            st.markdown("""
                <style>
                [data-testid="stTextArea"] { max-width: 750px; }
                </style>
            """, unsafe_allow_html=True)
        user_query = st.text_area(
            "Enter your question",
            placeholder="e.g. How many total products are there?",
            height=68,
        )

        if st.button("🚀 Analyze"):
            if not st.session_state.file_path or not user_query:
                st.warning("Please upload/select an image and enter a question.")
                return

            total_start = time.time()
            timings: Dict[str, float] = {}

            with st.spinner("Processing…"):
                if not st.session_state.image_processed:
                    # Determine blob_name from the active file (may be a local path)
                    active_file = st.session_state.file_path
                    _active_blob_name = Path(active_file).name if active_file else None

                    annotated_paths, row_counts, error, vision_timings = process_image_once(
                        image_path=active_file,
                        dirs=dirs,
                        blob_name=_active_blob_name,
                        conn_str=s["AZURE_BLOB_CONN"],
                        container=s["AZURE_BLOB_CONTAINER"],
                    )
                    timings.update(vision_timings)
                    if error:
                        st.error(error)
                        return
                    st.session_state.annotated_paths = annotated_paths
                    st.session_state.row_counts = row_counts
                    st.session_state.image_processed = True

                llm_start = time.time()
                try:
                    result = llm.run(
                        user_query=user_query,
                        image_paths=st.session_state.annotated_paths or [],
                        clear_history=True,
                    )
                except Exception as e:
                    st.error(f"LLM failed: {e}")
                    return

                timings["LLM processing"] = time.time() - llm_start
                timings["Total end-to-end"] = time.time() - total_start
                st.session_state.latency_stats = timings
                st.session_state.final_answer = result.get("final_answer", "")
                st.session_state.final_reasoning = result.get("reasoning", "")

        # ── Results ───────────────────────────────────────────────
        if st.session_state.final_answer:
            st.markdown("### 📝 Final Answer")
            show_reasoning = st.toggle("🧠 Show Reasoning", value=False)
            st.markdown(f"**📌 Answer:** {st.session_state.final_answer}")
            if show_reasoning:
                st.markdown(f"**🧠 Reasoning:** {st.session_state.final_reasoning}")

            with st.expander("💬 Chat History"):
                for idx, msg in enumerate(llm.manager.groupchat.messages):
                    st.markdown(f"### {idx+1}: {msg.get('name')}")
                    content = msg.get("content", "")
                    try:
                        st.json(json.loads(content))
                    except Exception:
                        st.code(content)

            if st.session_state.latency_stats:
                with st.expander("⏱️ Latency Breakdown"):
                    for k, v in st.session_state.latency_stats.items():
                        st.write(f"{k}: {v:.3f} sec")

            if st.session_state.image_processed:
                with st.expander("🖼️ Image Processing", expanded=False):
                    st.markdown("### 1️⃣ Rows Summary")
                    if dirs["SUMMARY_OUT"].exists():
                        st.image(str(dirs["SUMMARY_OUT"]), width=450)

                    st.markdown("### 2️⃣ Cropped Rows")
                    cropped_files = sorted(dirs["ROW_DIR"].glob("*.png"))
                    if cropped_files:
                        cols = st.columns(min(4, len(cropped_files)))
                        for i, p in enumerate(cropped_files):
                            with cols[i % len(cols)]:
                                st.image(str(p), use_container_width=True)

                    st.markdown("### 3️⃣ Product Detection Rows")
                    annotated_files = sorted(dirs["ANNOTATED_DIR"].glob("*.png"))
                    if annotated_files:
                        cols = st.columns(min(4, len(annotated_files)))
                        for i, p in enumerate(annotated_files):
                            with cols[i % len(cols)]:
                                st.image(str(p), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # RIGHT PANEL — Browse + Semantic Search
    # ═══════════════════════════════════════════════════════════════
    if mode == "📁 Select From Blob" and right_col is not None:
        with right_col:

            if st.session_state.file_path:
                fname = Path(st.session_state.file_path).name
                st.markdown(
                    f'<div class="sel-badge">✓ {fname}</div>',
                    unsafe_allow_html=True,
                )

            tabs = st.tabs(["🗂️ Browse", "🔎 Search"])

            # ────────────────────────────────────────────────────
            # TAB 1 — Browse (paginated)
            # ────────────────────────────────────────────────────
            with tabs[0]:
                # Re-fetch blob list here so newly uploaded images are included
                blob_files = list_blobs_images_cached(s["AZURE_BLOB_CONN"], s["AZURE_BLOB_CONTAINER"])

                BROWSE_PAGE_SIZE = 20
                total_blobs = len(blob_files)
                total_pages = max(1, (total_blobs + BROWSE_PAGE_SIZE - 1) // BROWSE_PAGE_SIZE)

                st.session_state.browse_page = min(st.session_state.browse_page, total_pages - 1)
                page = st.session_state.browse_page

                pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
                with pcol1:
                    if st.button("◀", disabled=(page == 0), key="browse_prev", help="Previous page"):
                        st.session_state.browse_page -= 1
                        st.rerun()
                with pcol2:
                    st.caption(f"Page {page + 1} / {total_pages}  ·  {total_blobs} images")
                with pcol3:
                    if st.button("▶", disabled=(page >= total_pages - 1), key="browse_next", help="Next page"):
                        st.session_state.browse_page += 1
                        st.rerun()

                page_start = page * BROWSE_PAGE_SIZE
                page_blobs = blob_files[page_start: page_start + BROWSE_PAGE_SIZE]

                items = []
                for blob_name in page_blobs:
                    try:
                        sas_url = get_blob_sas_url_cached(
                            s["AZURE_BLOB_CONN"], s["AZURE_BLOB_CONTAINER"], blob_name
                        )
                        items.append({
                            "blob_name": blob_name,
                            "container": s["AZURE_BLOB_CONTAINER"],
                            "sas_url": sas_url,
                        })
                    except Exception:
                        pass

                st.markdown('<div class="gallery-scroll">', unsafe_allow_html=True)
                if render_gallery(items, st.session_state.file_path or "", upload_dir, s, key_prefix="brw"):
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # ────────────────────────────────────────────────────
            # TAB 2 — Semantic Search + GPT-4o Validation
            # ────────────────────────────────────────────────────
            with tabs[1]:
                q = st.text_input(
                    "Describe the shelf image you want",
                    key="vector_query_input",
                    placeholder="e.g. chips aisle with blue bags",
                )
                top_k = 10
                use_gpt_validation = True

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    do_search = st.button(
                        "🔎 Search", key="vector_search_button", use_container_width=True
                    )
                with col_b:
                    if st.button("✕ Clear", key="vector_clear_button", use_container_width=True):
                        st.session_state.vector_results = []
                        st.session_state.vector_raw_count = None
                        st.rerun()

                # ── Run search pipeline ────────────────────────
                if do_search:
                    if not q.strip():
                        st.warning("Enter a search query.")
                    else:
                        raw_candidates: List[Dict] = []

                        with st.spinner(f"🔍 Retrieving top {top_k} candidates…"):
                            try:
                                raw_candidates = vector_search_blobs(q.strip(), int(top_k))
                                st.session_state.vector_query_text = q.strip()
                                st.session_state.vector_top_k = int(top_k)
                                st.session_state.vector_raw_count = len(raw_candidates)
                            except Exception as e:
                                st.error(f"Vector search failed: {e}")
                                st.session_state.vector_results = []
                                st.session_state.vector_raw_count = 0

                        if raw_candidates:
                            if use_gpt_validation:
                                progress_ph = st.empty()
                                progress_ph.caption(
                                    f"🤖 validating {len(raw_candidates)} images…"
                                )
                                try:
                                    validated = validate_search_results_with_gpt(
                                        candidates=raw_candidates,
                                        user_query=q.strip(),
                                        conn_str=s["AZURE_BLOB_CONN"],
                                        progress_placeholder=progress_ph,
                                    )
                                    st.session_state.vector_results = validated
                                except Exception as e:
                                    st.error(f"Validation error: {e}")
                                    st.session_state.vector_results = raw_candidates
                            else:
                                st.session_state.vector_results = raw_candidates

                # ── Display results ────────────────────────────
                results: List[Dict] = st.session_state.vector_results or []
                raw_count: Optional[int] = st.session_state.vector_raw_count

                if results:
                    n_shown = len(results)
                    query_label = st.session_state.vector_query_text or ""

                    if raw_count is not None and use_gpt_validation:
                        st.caption(f"")
                    else:
                        st.caption(f"Top {n_shown} results  ·  \"{query_label}\"")

                    vec_items: List[Dict] = []
                    for item in results:
                        blob_name = item.get("blob_name")
                        cont = item.get("container") or s["AZURE_BLOB_CONTAINER"]
                        if not blob_name:
                            continue
                        try:
                            sas_url = get_blob_sas_url_cached(s["AZURE_BLOB_CONN"], cont, blob_name)
                            vec_items.append({
                                "blob_name": blob_name,
                                "container": cont,
                                "sas_url": sas_url,
                                "gpt_reason": item.get("gpt_reason", ""),
                                "gpt_validated": item.get("gpt_validated", False),
                            })
                        except Exception:
                            pass

                    st.markdown('<div class="gallery-scroll">', unsafe_allow_html=True)
                    if render_gallery(
                        vec_items,
                        st.session_state.file_path or "",
                        upload_dir,
                        s,
                        key_prefix="vec",
                        show_gpt_reason=use_gpt_validation,
                    ):
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                elif raw_count is not None and raw_count > 0:
                    st.warning(
                        f"🤖 GPT-4o found no relevant images among {raw_count} candidates. "
                        "Try a different query or disable validation."
                    )
                elif raw_count == 0:
                    st.info("No images found in the index for that query.")
                else:
                    st.markdown(
                        """
                        <div style="text-align:center; padding:40px 0; color:#6b7280; font-size:13px;">
                            🔎 Run a search to see matching images
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()