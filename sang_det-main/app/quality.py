"""Geometry filters and crop quality gates.

Two stages, cheapest first:

  1. `passes_geometry` runs on the bounding box before any pixels are copied -
     size, aspect ratio and area sanity. This is what rejects tiny unreadable
     boxes and whole-vehicle false positives.
  2. `assess` runs on the crop itself - Laplacian-variance blur detection,
     contrast, and a flat-region check for heavy glare or occlusion.

Anything rejected here never reaches the output folder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .detector import Detection


@dataclass
class QualityReport:
    accepted: bool
    reason: str = ""
    blur_score: float = 0.0
    contrast: float = 0.0
    uniform_fraction: float = 0.0


def passes_geometry(
    det: Detection, frame_w: int, frame_h: int, cfg: Config
) -> tuple[bool, str]:
    """Box-level filters. Cheap, so they run before cropping."""
    min_w = int(cfg.get("filters.min_box_width", 60))
    min_h = int(cfg.get("filters.min_box_height", 20))
    if det.width < min_w or det.height < min_h:
        return False, f"too_small({det.width}x{det.height})"

    ratio = det.aspect_ratio
    min_ar = float(cfg.get("filters.min_aspect_ratio", 1.2))
    max_ar = float(cfg.get("filters.max_aspect_ratio", 8.0))
    if ratio < min_ar or ratio > max_ar:
        return False, f"aspect({ratio:.2f})"

    frame_area = max(1, frame_w * frame_h)
    max_fraction = float(cfg.get("filters.max_area_fraction", 0.25))
    if det.area / frame_area > max_fraction:
        return False, f"too_large({det.area / frame_area:.2f})"

    return True, ""


def crop(image: np.ndarray, det: Detection, cfg: Config) -> np.ndarray | None:
    """Tight crop of the plate region only, with a few px of context.

    Never returns a full frame or a vehicle crop - the box comes straight from
    the plate detector and padding is a handful of pixels.
    """
    pad = max(0, int(cfg.get("filters.crop_padding_px", 3)))
    height, width = image.shape[:2]

    x1 = max(0, det.x1 - pad)
    y1 = max(0, det.y1 - pad)
    x2 = min(width, det.x2 + pad)
    y2 = min(height, det.y2 + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    region = image[y1:y2, x1:x2]
    return np.ascontiguousarray(region) if region.size else None


def assess(crop_bgr: np.ndarray, cfg: Config) -> QualityReport:
    """Blur / contrast / occlusion gate on the crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return QualityReport(False, "empty_crop")

    height, width = crop_bgr.shape[:2]
    min_pixels = int(cfg.get("quality.min_crop_pixels", 1200))
    if height * width < min_pixels:
        return QualityReport(False, f"crop_pixels({height * width})")

    if not bool(cfg.get("quality.enabled", True)):
        return QualityReport(True)

    try:
        import cv2

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        gray = crop_bgr.mean(axis=2).astype(np.float64)
        # Discrete Laplacian without OpenCV, same idea, slower.
        laplace = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
        )
        blur_score = float(laplace.var()) if laplace.size else 0.0
        gray = gray.astype(np.uint8)

    min_blur = float(cfg.get("quality.min_laplacian_variance", 45.0))
    if blur_score < min_blur:
        return QualityReport(False, f"blurred({blur_score:.1f})", blur_score=blur_score)

    contrast = float(np.std(gray))
    min_contrast = float(cfg.get("quality.min_contrast_stddev", 18.0))
    if contrast < min_contrast:
        return QualityReport(
            False, f"low_contrast({contrast:.1f})",
            blur_score=blur_score, contrast=contrast,
        )

    # A crop that is overwhelmingly one intensity is glare, shadow, or an
    # occluding object - not a readable plate.
    histogram = np.bincount(np.asarray(gray).ravel(), minlength=256)
    uniform_fraction = float(histogram.max() / max(1, gray.size))
    max_uniform = float(cfg.get("quality.max_uniform_fraction", 0.92))
    if uniform_fraction > max_uniform:
        return QualityReport(
            False, f"uniform({uniform_fraction:.2f})",
            blur_score=blur_score, contrast=contrast,
            uniform_fraction=uniform_fraction,
        )

    return QualityReport(
        True, blur_score=blur_score, contrast=contrast,
        uniform_fraction=uniform_fraction,
    )
