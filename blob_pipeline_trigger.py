"""
blob_pipeline_trigger.py

Runs the Roboflow vision pipeline immediately when an image is uploaded to
Azure Blob Storage, and persists results back into the same container under
a structured folder hierarchy:

  <container>/
    <image_stem>/                        ← folder named after the image (no extension)
      original/<image_filename>          ← original image (already uploaded by caller)
      cropped_images/row_1.png           ← masked row crops from detect_and_crop_rows
      cropped_images/row_2.png
      cropped_images/shelf_summary.jpg   ← polygon-overlay summary image
      products_detected/row_1.png        ← annotated product-detection images
      products_detected/row_2.png
      metadata.json                      ← row_counts + pipeline status

At query time the Streamlit app fetches these pre-processed images from blob
instead of re-running Roboflow, saving significant latency per query.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from azure.storage.blob import BlobServiceClient

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BLOB PATH CONVENTIONS
# ─────────────────────────────────────────────────────────────────────────────
# One shared BlobServiceClient per connection string — avoids recreating on every call
_bsc_cache: Dict[str, BlobServiceClient] = {}
_bsc_lock = threading.Lock()

def _get_bsc(conn_str: str) -> BlobServiceClient:
    """Returns a cached BlobServiceClient for the given connection string."""
    if conn_str not in _bsc_cache:
        with _bsc_lock:
            if conn_str not in _bsc_cache:
                _bsc_cache[conn_str] = BlobServiceClient.from_connection_string(conn_str)
    return _bsc_cache[conn_str]

def _image_stem(blob_name: str) -> str:
    """
    Returns the 'folder name' for a blob, derived from the blob's filename stem.
    e.g. 'uploads/shelf_photo.jpg' → 'shelf_photo'
         'shelf_photo.png'         → 'shelf_photo'
    """
    return Path(blob_name).stem


def processed_prefix(blob_name: str) -> str:
    """Root folder in blob for all pipeline outputs of a given image."""
    return f"{_image_stem(blob_name)}/"


def cropped_prefix(blob_name: str) -> str:
    return f"{_image_stem(blob_name)}/cropped_images/"


def products_prefix(blob_name: str) -> str:
    return f"{_image_stem(blob_name)}/products_detected/"


def metadata_blob_name(blob_name: str) -> str:
    return f"{_image_stem(blob_name)}/metadata.json"


def summary_blob_name(blob_name: str) -> str:
    return f"{_image_stem(blob_name)}/cropped_images/shelf_summary.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# BLOB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# def _get_container_client(conn_str: str, container: str):
#     bsc = BlobServiceClient.from_connection_string(conn_str)
#     return bsc.get_container_client(container)

def _get_container_client(conn_str: str, container: str):
    return _get_bsc(conn_str).get_container_client(container)

def _upload_bytes_direct(conn_str: str, container: str, blob_path: str,
                          data: bytes, content_type: str = "image/png") -> None:
    """Upload raw bytes directly — no file open/close overhead."""
    from azure.storage.blob import ContentSettings
    cc = _get_container_client(conn_str, container)
    cc.upload_blob(
        name=blob_path,
        data=data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    logger.debug("Uploaded → %s/%s", container, blob_path)

# def _upload_file_to_blob(
#     conn_str: str,
#     container: str,
#     blob_path: str,
#     local_file: str,
#     content_type: str = "image/png",
# ) -> None:
#     """Uploads a local file to blob at the given blob_path."""
#     cc = _get_container_client(conn_str, container)
#     with open(local_file, "rb") as f:
#         cc.upload_blob(
#             name=blob_path,
#             data=f,
#             overwrite=True,
#             content_settings=_content_settings(content_type),
#         )
#     logger.debug("Uploaded → %s/%s", container, blob_path)
def _upload_file_to_blob(conn_str: str, container: str, blob_path: str,
                          local_file: str, content_type: str = "image/png") -> None:
    """Read file bytes once and upload directly."""
    data = Path(local_file).read_bytes()
    _upload_bytes_direct(conn_str, container, blob_path, data, content_type)


def _upload_bytes_to_blob(
    conn_str: str,
    container: str,
    blob_path: str,
    data: bytes,
    content_type: str = "application/json",
) -> None:
    cc = _get_container_client(conn_str, container)
    cc.upload_blob(name=blob_path, data=data, overwrite=True,
                   content_settings=_content_settings(content_type))


def _content_settings(content_type: str):
    from azure.storage.blob import ContentSettings
    return ContentSettings(content_type=content_type)


# def _download_blob_to_file(
#     conn_str: str,
#     container: str,
#     blob_path: str,
#     local_file: str,
# ) -> None:
#     cc = _get_container_client(conn_str, container)
#     bc = cc.get_blob_client(blob_path)
#     with open(local_file, "wb") as f:
#         f.write(bc.download_blob().readall())
def _download_blob_to_file(conn_str: str, container: str,
                            blob_path: str, local_file: str) -> None:
    cc = _get_container_client(conn_str, container)
    data = cc.get_blob_client(blob_path).download_blob().readall()
    Path(local_file).write_bytes(data)

# def _blob_exists(conn_str: str, container: str, blob_path: str) -> bool:
#     try:
#         cc = _get_container_client(conn_str, container)
#         bc = cc.get_blob_client(blob_path)
#         bc.get_blob_properties()
#         return True
#     except Exception:
#         return False
def _blob_exists(conn_str: str, container: str, blob_path: str) -> bool:
    try:
        _get_container_client(conn_str, container).get_blob_client(blob_path).get_blob_properties()
        return True
    except Exception:
        return False

def _list_blobs_with_prefix(conn_str: str, container: str, prefix: str) -> List[str]:
    cc = _get_container_client(conn_str, container)
    return [b.name for b in cc.list_blobs(name_starts_with=prefix)]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_already_processed(conn_str: str, container: str, blob_name: str) -> bool:
    """
    Returns True if metadata.json exists for this image — meaning the
    Roboflow pipeline has already been run and its outputs are in blob.
    """
    return _blob_exists(conn_str, container, metadata_blob_name(blob_name))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRIGGER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

# def run_pipeline_and_store(
#     conn_str: str,
#     container: str,
#     blob_name: str,
#     local_image_path: str,
#     force: bool = False,
# ) -> Tuple[bool, Optional[str]]:
#     """
#     Runs the full Roboflow vision pipeline for `blob_name` and stores all
#     outputs back into blob storage.

#     This function is designed to be called right after an image is uploaded.
#     It is idempotent: if outputs already exist it returns early (unless force=True).

#     Args:
#         conn_str:         Azure Blob Storage connection string.
#         container:        Container name.
#         blob_name:        Blob name of the original image (e.g. 'shelf_photo.jpg').
#         local_image_path: Local path where the original image has been saved.
#         force:            Re-run pipeline even if outputs already exist.

#     Returns:
#         (success: bool, error_message: Optional[str])
#     """
#     if not force and is_already_processed(conn_str, container, blob_name):
#         logger.info("Pipeline outputs already exist for '%s'. Skipping.", blob_name)
#         return True, None

#     logger.info("Running Roboflow pipeline for '%s'…", blob_name)

#     # Import here to avoid circular imports and Streamlit caching issues
#     from detect_crop import detect_and_crop_rows, annotate_rows_with_yolo

#     with tempfile.TemporaryDirectory(prefix="roboflow_pipeline_") as tmpdir:
#         tmp = Path(tmpdir)
#         crop_dir = tmp / "cropped_images"
#         annotated_dir = tmp / "products_detected"
#         summary_out = tmp / "shelf_summary.jpg"
#         crop_dir.mkdir()
#         annotated_dir.mkdir()

#         # ── Step 0: Upload original image under <stem>/original/ ──────
#         original_ext = Path(blob_name).suffix.lower()
#         original_ctype = "image/jpeg" if original_ext in (".jpg", ".jpeg") else "image/png"
#         original_blob_path = f"{_image_stem(blob_name)}/original/{Path(blob_name).name}"
#         _upload_file_to_blob(
#             conn_str, container, original_blob_path, local_image_path, content_type=original_ctype
#         )

#         # ── Step 1: Row detection + cropping ──────────────────────────
#         row_paths, error = detect_and_crop_rows(
#             image_path=local_image_path,
#             summary_out=str(summary_out),
#             crop_dir=str(crop_dir),
#         )

#         if error:
#             _store_metadata(
#                 conn_str, container, blob_name,
#                 status="error",
#                 error=error,
#                 row_counts=[],
#             )
#             return False, error

#         if not row_paths:
#             msg = "Row detection produced no crops."
#             _store_metadata(
#                 conn_str, container, blob_name,
#                 status="error",
#                 error=msg,
#                 row_counts=[],
#             )
#             return False, msg

#         # ── Step 2: Product detection + annotation ─────────────────────
#         annotated_paths, row_counts = annotate_rows_with_yolo(
#             row_paths=row_paths,
#             output_dir=str(annotated_dir),
#         )

#         # ── Steps 3-6: Upload all outputs in parallel ──────────────────
#         from concurrent.futures import ThreadPoolExecutor, as_completed

#         upload_tasks: List[Tuple[str, str, str]] = []  # (local_path, blob_dest, content_type)

#         # Summary image
#         if summary_out.exists():
#             upload_tasks.append((
#                 str(summary_out),
#                 summary_blob_name(blob_name),
#                 "image/jpeg",
#             ))

#         # Cropped row images
#         for crop_path in sorted(crop_dir.glob("row_*.png")):
#             upload_tasks.append((
#                 str(crop_path),
#                 f"{_image_stem(blob_name)}/cropped_images/{crop_path.name}",
#                 "image/png",
#             ))

#         # Annotated product-detection images
#         for ann_path in sorted(annotated_dir.glob("row_*.png")):
#             upload_tasks.append((
#                 str(ann_path),
#                 f"{_image_stem(blob_name)}/products_detected/{ann_path.name}",
#                 "image/png",
#             ))

#         def _upload_task(task: Tuple[str, str, str]):
#             local_f, blob_dest, ctype = task
#             _upload_file_to_blob(conn_str, container, blob_dest, local_f, content_type=ctype)
#             logger.debug("Uploaded → %s", blob_dest)

#         upload_errors = []
#         with ThreadPoolExecutor(max_workers=min(len(upload_tasks), 10)) as executor:
#             futures = {executor.submit(_upload_task, t): t for t in upload_tasks}
#             for future in as_completed(futures):
#                 try:
#                     future.result()
#                 except Exception as e:
#                     upload_errors.append(str(e))
#                     logger.error("Upload failed: %s", e)

#         if upload_errors:
#             logger.warning("%d upload(s) failed: %s", len(upload_errors), upload_errors)

#         # Metadata written last — only after all images are up
#         _store_metadata(
#             conn_str, container, blob_name,
#             status="success" if not upload_errors else "partial",
#             error="; ".join(upload_errors) if upload_errors else None,
#             row_counts=row_counts,
#         )

#     logger.info("Pipeline complete for '%s'. Outputs stored in blob.", blob_name)
#     return True, None
def run_pipeline_and_store(
    conn_str: str,
    container: str,
    blob_name: str,
    local_image_path: str,
    force: bool = False,
) -> Tuple[bool, Optional[str]]:
    if not force and is_already_processed(conn_str, container, blob_name):
        logger.info("Pipeline outputs already exist for '%s'. Skipping.", blob_name)
        return True, None

    logger.info("Running Roboflow pipeline for '%s'…", blob_name)
    from detect_crop import detect_and_crop_rows, annotate_rows_with_yolo

    with tempfile.TemporaryDirectory(prefix="roboflow_pipeline_") as tmpdir:
        tmp = Path(tmpdir)
        crop_dir = tmp / "cropped_images"
        annotated_dir = tmp / "products_detected"
        summary_out = tmp / "shelf_summary.jpg"
        crop_dir.mkdir()
        annotated_dir.mkdir()

        # Step 1: Row detection + cropping (must be sequential — depends on full image)
        row_paths, error = detect_and_crop_rows(
            image_path=local_image_path,
            summary_out=str(summary_out),
            crop_dir=str(crop_dir),
        )

        if error:
            _store_metadata(conn_str, container, blob_name,
                            status="error", error=error, row_counts=[])
            return False, error

        if not row_paths:
            msg = "Row detection produced no crops."
            _store_metadata(conn_str, container, blob_name,
                            status="error", error=msg, row_counts=[])
            return False, msg

        # Step 2: Product detection — run all rows in parallel
        annotated_paths, row_counts = annotate_rows_with_yolo(
            row_paths=row_paths, output_dir=str(annotated_dir)
        )

        # Step 3: Collect ALL files to upload (original + summary + crops + annotated)
        stem = _image_stem(blob_name)
        original_ext = Path(blob_name).suffix.lower()
        original_ctype = "image/jpeg" if original_ext in (".jpg", ".jpeg") else "image/png"

        # (local_path, blob_dest, content_type)
        upload_tasks: List[Tuple[str, str, str]] = []

        upload_tasks.append((
            local_image_path,
            f"{stem}/original/{Path(blob_name).name}",
            original_ctype,
        ))

        if summary_out.exists():
            upload_tasks.append((str(summary_out), summary_blob_name(blob_name), "image/jpeg"))

        for p in sorted(crop_dir.glob("row_*.png")):
            upload_tasks.append((str(p), f"{stem}/cropped_images/{p.name}", "image/png"))

        for p in sorted(annotated_dir.glob("row_*.png")):
            upload_tasks.append((str(p), f"{stem}/products_detected/{p.name}", "image/png"))

        # Step 4: Upload ALL files in parallel
        def _do_upload(task: Tuple[str, str, str]):
            local_f, blob_dest, ctype = task
            _upload_file_to_blob(conn_str, container, blob_dest, local_f, content_type=ctype)

        upload_errors = []
        with ThreadPoolExecutor(max_workers=min(len(upload_tasks), 12)) as ex:
            futures = {ex.submit(_do_upload, t): t for t in upload_tasks}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    upload_errors.append(str(e))
                    logger.error("Upload failed: %s", e)

        # Step 5: Write metadata last — it acts as the "done" signal
        _store_metadata(
            conn_str, container, blob_name,
            status="success" if not upload_errors else "partial",
            error="; ".join(upload_errors) if upload_errors else None,
            row_counts=row_counts,
        )

    logger.info("Pipeline complete for '%s'.", blob_name)
    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# FETCH PRE-PROCESSED RESULTS FROM BLOB
# ─────────────────────────────────────────────────────────────────────────────

# def fetch_processed_images(
#     conn_str: str,
#     container: str,
#     blob_name: str,
#     local_dir: str,
#     row_dir: Optional[str] = None,
#     annotated_dir: Optional[str] = None,
# ) -> Tuple[List[str], List[str], List[Dict]]:
#     """
#     Downloads all pre-processed pipeline outputs from blob to local directories.

#     Args:
#         local_dir:      Root temp dir (for summary image).
#         row_dir:        Where to save cropped row images. Defaults to local_dir/cropped_images.
#         annotated_dir:  Where to save annotated product images. Defaults to local_dir/products_detected.
#     """
#     local = Path(local_dir)
#     stem = _image_stem(blob_name)

#     crop_local = Path(row_dir) if row_dir else local / "cropped_images"
#     ann_local = Path(annotated_dir) if annotated_dir else local / "products_detected"
#     crop_local.mkdir(parents=True, exist_ok=True)
#     ann_local.mkdir(parents=True, exist_ok=True)

#     # ── Summary ─────────────────────────────────────────────────────────
#     # Downloaded directly to local/shelf_summary.jpg, which is the same
#     # path as dirs["SUMMARY_OUT"] in the caller — no copy needed.
#     s_blob = summary_blob_name(blob_name)
#     if _blob_exists(conn_str, container, s_blob):
#         s_local = str(local / "shelf_summary.jpg")
#         _download_blob_to_file(conn_str, container, s_blob, s_local)

#     # ── Cropped rows ───────────────────────────────────────────────────
#     cropped_paths: List[str] = []
#     for b in sorted(_list_blobs_with_prefix(conn_str, container, f"{stem}/cropped_images/row_")):
#         fname = Path(b).name
#         local_f = str(crop_local / fname)
#         _download_blob_to_file(conn_str, container, b, local_f)
#         cropped_paths.append(local_f)

#     # ── Annotated rows ─────────────────────────────────────────────────
#     annotated_paths: List[str] = []
#     for b in sorted(_list_blobs_with_prefix(conn_str, container, f"{stem}/products_detected/row_")):
#         fname = Path(b).name
#         local_f = str(ann_local / fname)
#         _download_blob_to_file(conn_str, container, b, local_f)
#         annotated_paths.append(local_f)

#     # ── Metadata / row counts ──────────────────────────────────────────
#     row_counts: List[Dict] = []
#     meta_blob = metadata_blob_name(blob_name)
#     if _blob_exists(conn_str, container, meta_blob):
#         cc = _get_container_client(conn_str, container)
#         raw = cc.get_blob_client(meta_blob).download_blob().readall()
#         meta = json.loads(raw)
#         row_counts = meta.get("row_counts", [])

#     return cropped_paths, annotated_paths, row_counts
def fetch_processed_images(
    conn_str: str,
    container: str,
    blob_name: str,
    local_dir: str,
    row_dir: Optional[str] = None,
    annotated_dir: Optional[str] = None,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Downloads all pre-processed pipeline outputs from blob in parallel using threads.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    local = Path(local_dir)
    stem = _image_stem(blob_name)

    crop_local = Path(row_dir) if row_dir else local / "cropped_images"
    ann_local = Path(annotated_dir) if annotated_dir else local / "products_detected"
    crop_local.mkdir(parents=True, exist_ok=True)
    ann_local.mkdir(parents=True, exist_ok=True)

    # ── List all blobs needed (3 prefix calls, can't easily parallelise listing) ──
    cropped_blobs = sorted(_list_blobs_with_prefix(conn_str, container, f"{stem}/cropped_images/row_"))
    annotated_blobs = sorted(_list_blobs_with_prefix(conn_str, container, f"{stem}/products_detected/row_"))
    summary_blob = summary_blob_name(blob_name)
    meta_blob = metadata_blob_name(blob_name)

    # ── Build download task list ───────────────────────────────────────
    # Each task: (blob_path, local_file_path)
    tasks: List[Tuple[str, str]] = []

    tasks.append((summary_blob, str(local / "shelf_summary.jpg")))

    cropped_local_map: Dict[str, str] = {}
    for b in cropped_blobs:
        local_f = str(crop_local / Path(b).name)
        tasks.append((b, local_f))
        cropped_local_map[b] = local_f

    annotated_local_map: Dict[str, str] = {}
    for b in annotated_blobs:
        local_f = str(ann_local / Path(b).name)
        tasks.append((b, local_f))
        annotated_local_map[b] = local_f

    tasks.append((meta_blob, str(local / "metadata.json")))

    # ── Download all in parallel ───────────────────────────────────────
    def _download(task: Tuple[str, str]):
        b_path, l_path = task
        try:
            _download_blob_to_file(conn_str, container, b_path, l_path)
            return b_path, True
        except Exception as e:
            logger.warning(f"Download failed for '{b_path}': {e}")
            return b_path, False

    with ThreadPoolExecutor(max_workers=min(len(tasks), 12)) as executor:
        futures = {executor.submit(_download, t): t for t in tasks}
        results_map = {b: ok for b, ok in (f.result() for f in as_completed(futures))}

    # ── Build ordered return lists ─────────────────────────────────────
    cropped_paths = [
        cropped_local_map[b] for b in cropped_blobs
        if results_map.get(b) and Path(cropped_local_map[b]).exists()
    ]
    annotated_paths = [
        annotated_local_map[b] for b in annotated_blobs
        if results_map.get(b) and Path(annotated_local_map[b]).exists()
    ]

    # ── Read metadata ──────────────────────────────────────────────────
    row_counts: List[Dict] = []
    meta_local = local / "metadata.json"
    if meta_local.exists():
        try:
            row_counts = json.loads(meta_local.read_text()).get("row_counts", [])
        except Exception:
            pass

    return cropped_paths, annotated_paths, row_counts

# ─────────────────────────────────────────────────────────────────────────────
# METADATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _store_metadata(
    conn_str: str,
    container: str,
    blob_name: str,
    status: str,
    error: Optional[str],
    row_counts: List[Dict],
) -> None:
    import datetime
    payload = {
        "blob_name": blob_name,
        "status": status,
        "error": error,
        "row_counts": row_counts,
        "processed_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    _upload_bytes_to_blob(
        conn_str, container,
        metadata_blob_name(blob_name),
        json.dumps(payload, indent=2).encode("utf-8"),
        content_type="application/json",
    )


def get_pipeline_metadata(conn_str: str, container: str, blob_name: str) -> Optional[Dict]:
    """
    Returns the metadata dict for a processed image, or None if not found.
    Useful for checking pipeline status before querying.
    """
    meta_blob = metadata_blob_name(blob_name)
    if not _blob_exists(conn_str, container, meta_blob):
        return None
    cc = _get_container_client(conn_str, container)
    raw = cc.get_blob_client(meta_blob).download_blob().readall()
    return json.loads(raw)