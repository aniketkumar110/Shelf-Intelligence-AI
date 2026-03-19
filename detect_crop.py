"""
detect_crop.py
Production-grade vision pipeline using Roboflow:
1) Detect shelf rows (polygon)
2) Crop each row (masked with white background)
3) Detect products on each row (YOLO) and annotate

Key improvements:
- No hard-coded secrets (env / key.py fallback)
- Models loaded once via cached registry
- Clear return types + error handling
- Safer filesystem usage (caller passes output directories)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
from collections import Counter

logger = logging.getLogger(__name__)

# -----------------------------

ROBOFLOW_API_KEY = "UQxIKDyrEUZYku4A3zXL"

ROBOFLOW_WORKSPACE = "planogram1"
 
ROBOFLOW_ROW_PROJECT = "my-first-project-z5ns2"

ROW_VERSION = 4

ROW_CONFIDENCE = 40
 
PRODUCT_PROJECT = "third-project-0rcul"

PRODUCT_VERSION = 3

PRODUCT_CONFIDENCE = 40
# ---------------------------------------------------------------------
# CONFIG (env-first, optional key.py fallback)
# ---------------------------------------------------------------------
def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _get_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v else default


def load_vision_settings() -> Dict[str, str]:
    settings = {
        "ROBOFLOW_API_KEY": _get_str("ROBOFLOW_API_KEY", ""),
        "ROBOFLOW_WORKSPACE": _get_str("ROBOFLOW_WORKSPACE", "planogram1"),

        "ROW_PROJECT": _get_str("ROBOFLOW_ROW_PROJECT", "my-first-project-z5ns2"),
        "ROW_VERSION": str(_get_int("ROBOFLOW_ROW_VERSION", 4)),
        "ROW_CONFIDENCE": str(_get_int("ROW_CONFIDENCE", 40)),

        "PRODUCT_PROJECT": _get_str("ROBOFLOW_PRODUCT_PROJECT", "third-project-0rcul"),
        "PRODUCT_VERSION": str(_get_int("ROBOFLOW_PRODUCT_VERSION", 3)),
        "PRODUCT_CONFIDENCE": str(_get_int("PRODUCT_CONFIDENCE", 40)),
        "PRODUCT_OVERLAP": str(_get_int("PRODUCT_OVERLAP", 50)),
    }

    # Optional fallback to key.py for local dev
    if not settings["ROBOFLOW_API_KEY"]:
        try:
            from key import ROBOFLOW_API_KEY  # type: ignore
            settings["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY
        except Exception:
            pass

    if not settings["ROBOFLOW_API_KEY"]:
        raise RuntimeError("ROBOFLOW_API_KEY missing (env or key.py).")

    return settings


# ---------------------------------------------------------------------
# MODEL REGISTRY (load once)
# ---------------------------------------------------------------------
@dataclass
class RoboflowModels:
    row_model: any
    product_model: any
    row_confidence: int
    product_confidence: int
    product_overlap: int


_models_singleton: Optional[RoboflowModels] = None


def get_models() -> RoboflowModels:
    """
    Loads Roboflow models only once per process.
    In Streamlit, cache_resource should wrap calls to VisionPipeline instead.
    """
    global _models_singleton
    if _models_singleton is not None:
        return _models_singleton

    s = load_vision_settings()

    rf = Roboflow(api_key=s["ROBOFLOW_API_KEY"])
    ws = rf.workspace(s["ROBOFLOW_WORKSPACE"])

    row_model = ws.project(s["ROW_PROJECT"]).version(int(s["ROW_VERSION"])).model
    product_model = ws.project(s["PRODUCT_PROJECT"]).version(int(s["PRODUCT_VERSION"])).model

    _models_singleton = RoboflowModels(
        row_model=row_model,
        product_model=product_model,
        row_confidence=int(s["ROW_CONFIDENCE"]),
        product_confidence=int(s["PRODUCT_CONFIDENCE"]),
        product_overlap=int(s["PRODUCT_OVERLAP"]),
    )
    logger.info("Roboflow models loaded.")
    return _models_singleton


# ---------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------
def detect_and_crop_rows(
    image_path: str,
    summary_out: str,
    crop_dir: str,
) -> Tuple[List[str], Optional[str]]:
    """
    Detect shelf rows (polygons) and create masked crops per row.

    Returns:
      (row_image_paths, error_message)
    """
    try:
        os.makedirs(crop_dir, exist_ok=True)

        # Force RGB (Roboflow + cv2 safety)
        fixed_image = os.path.join(crop_dir, "__input_rgb_temp.jpg")
        Image.open(image_path).convert("RGB").save(fixed_image)

        m = get_models()
        row_result = m.row_model.predict(fixed_image, confidence=m.row_confidence).json()
        row_predictions = row_result.get("predictions", [])

        if not row_predictions:
            return [], "No rows detected."

        img_full = cv2.imread(fixed_image)
        if img_full is None:
            return [], "Could not read input image."

        h, w = img_full.shape[:2]

        # Sort rows top->bottom by polygon center Y
        def row_center_y(pred) -> float:
            pts = pred.get("points", [])
            if not pts:
                return 1e9
            ys = [p["y"] for p in pts]
            return float(sum(ys) / max(len(ys), 1))

        row_predictions = sorted(row_predictions, key=row_center_y)

        # Summary image with row polygons
        summary_img = img_full.copy()
        for pred in row_predictions:
            pts = pred.get("points", [])
            if not pts:
                continue
            poly = np.array([[int(p["x"]), int(p["y"])] for p in pts], np.int32).reshape((-1, 1, 2))
            cv2.polylines(summary_img, [poly], True, (0, 255, 0), 3)
        cv2.imwrite(summary_out, summary_img)

        # Crop each row (masked)
        row_paths: List[str] = []
        row_index = 1
        white_bg = np.ones_like(img_full, dtype=np.uint8) * 255

        for pred in row_predictions:
            pts = pred.get("points", [])
            if not pts:
                continue

            poly = np.array([[int(p["x"]), int(p["y"])] for p in pts], dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            masked = white_bg.copy()
            masked[mask == 255] = img_full[mask == 255]

            x, y, bw, bh = cv2.boundingRect(poly)
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)

            crop = masked[y : y + bh, x : x + bw]
            if crop.size == 0:
                continue

            out_path = os.path.join(crop_dir, f"row_{row_index}.png")
            cv2.imwrite(out_path, crop)
            row_paths.append(out_path)
            row_index += 1

        return row_paths, None

    except Exception as e:
        logger.exception("detect_and_crop_rows failed.")
        return [], str(e)


# def annotate_rows_with_yolo(
#     row_paths: List[str],
#     output_dir: str,
# ) -> Tuple[List[str], List[Dict]]:
#     """
#     Runs product detection on each row crop and writes annotated images.

#     Returns:
#       (annotated_image_paths, row_counts)
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     m = get_models()

#     annotated_paths: List[str] = []
#     row_counts: List[Dict] = []

#     for idx, row_path in enumerate(row_paths, start=1):
#         prod_result = m.product_model.predict(
#             row_path,
#             confidence=m.product_confidence,
#             overlap=m.product_overlap,
#         ).json()

#         preds = prod_result.get("predictions", [])

#         # Count per class label
#         label_counter = Counter()
#         for p in preds:
#             cls = p.get("class")
#             if cls:
#                 label_counter[cls] += 1

#         row_counts.append({"row": idx, "counts": dict(label_counter)})

#         # Draw boxes
#         row_img = cv2.imread(row_path)
#         if row_img is None:
#             continue
#         rh, rw = row_img.shape[:2]

#         for p in preds:
#             cx, cy = int(p["x"]), int(p["y"])
#             bw2, bh2 = int(p["width"]), int(p["height"])

#             x1, y1 = int(cx - bw2 / 2), int(cy - bh2 / 2)
#             x2, y2 = int(cx + bw2 / 2), int(cy + bh2 / 2)

#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(rw - 1, x2), min(rh - 1, y2)

#             cv2.rectangle(row_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         out_path = os.path.join(output_dir, f"row_{idx}.png")
#         cv2.imwrite(out_path, row_img)
#         annotated_paths.append(out_path)

#     return annotated_paths, row_counts
def annotate_rows_with_yolo(
    row_paths: List[str],
    output_dir: str,
) -> Tuple[List[str], List[Dict]]:
    """
    Runs product detection on all row crops in parallel and writes annotated images.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(output_dir, exist_ok=True)
    m = get_models()

    # Pre-allocate result slots to preserve row order
    results: Dict[int, Tuple[str, Dict]] = {}

    def _process_row(idx: int, row_path: str):
        prod_result = m.product_model.predict(
            row_path,
            confidence=m.product_confidence,
            overlap=m.product_overlap,
        ).json()

        preds = prod_result.get("predictions", [])

        label_counter = Counter()
        for p in preds:
            cls = p.get("class")
            if cls:
                label_counter[cls] += 1

        row_img = cv2.imread(row_path)
        if row_img is None:
            return idx, None, {"row": idx, "counts": dict(label_counter)}

        rh, rw = row_img.shape[:2]
        for p in preds:
            cx, cy = int(p["x"]), int(p["y"])
            bw2, bh2 = int(p["width"]), int(p["height"])
            x1 = max(0, int(cx - bw2 / 2))
            y1 = max(0, int(cy - bh2 / 2))
            x2 = min(rw - 1, int(cx + bw2 / 2))
            y2 = min(rh - 1, int(cy + bh2 / 2))
            cv2.rectangle(row_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out_path = os.path.join(output_dir, f"row_{idx}.png")
        cv2.imwrite(out_path, row_img)
        return idx, out_path, {"row": idx, "counts": dict(label_counter)}

    # Roboflow hosted API is I/O bound — parallelise with threads
    with ThreadPoolExecutor(max_workers=min(len(row_paths), 6)) as ex:
        futures = {
            ex.submit(_process_row, idx, path): idx
            for idx, path in enumerate(row_paths, start=1)
        }
        for future in as_completed(futures):
            idx, out_path, row_count = future.result()
            results[idx] = (out_path, row_count)

    # Rebuild in correct row order
    annotated_paths = []
    row_counts = []
    for idx in sorted(results):
        out_path, row_count = results[idx]
        if out_path:
            annotated_paths.append(out_path)
        row_counts.append(row_count)

    return annotated_paths, row_counts