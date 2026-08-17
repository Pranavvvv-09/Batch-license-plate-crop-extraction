"""Perceptual-hash deduplication.

A car passing a camera puts the same plate in ~10-40 consecutive sampled
frames. Saving all of them would flood the dataset with near-identical crops.

Every accepted crop is hashed and compared against a rolling window of recent
hashes; anything within `hash_distance` Hamming distance is a repeat and gets
dropped. Perceptual hashing (not OCR text) means this works on plates that are
too small or angled to read, and it tolerates the mild scale/lighting drift you
get as a vehicle approaches.

Two independent signals gate a match, because neither is sufficient alone:

  * **Visual** - Hamming distance between perceptual hashes. On its own this
    is a weak discriminator for plates specifically: every plate is the same
    rectangle in the same layout, and the part that differs (the characters)
    is a small fraction of the image that a perceptual hash largely discards.
    Measured on real detector output, same-plate and different-plate distance
    bands overlap, so there is no threshold that catches every repeat without
    also merging distinct plates.

  * **Temporal** - how far apart the two crops are in the source video. A
    vehicle passes in seconds, so two crops of the same plate are necessarily
    close in time. This is what makes an aggressive visual threshold safe: it
    stops a plate being compared against a similar-looking car from minutes
    ago, which is where false merges come from.

Defaults are deliberately conservative. Keeping a few extra near-duplicates
costs some dataset bloat and is trivially filtered later; wrongly merging two
distinct plates is unrecoverable data loss. When in doubt, this keeps the crop.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from .config import Config
from .logging_setup import get

log = get("dedup")

_ALGORITHMS = ("phash", "dhash", "ahash", "whash")


class Deduplicator:
    """Rolling-window near-duplicate filter over image hashes."""

    def __init__(self, cfg: Config):
        self._window: Deque[tuple[int, float, object]] = deque(maxlen=1)
        self._video_key = 0
        self._checked = 0
        self._dropped = 0
        self.configure(cfg)

    def configure(self, cfg: Config) -> None:
        """Re-read tunables. Safe to call between videos."""
        self.enabled = bool(cfg.get("dedup.enabled", True))
        self.distance = int(cfg.get("dedup.hash_distance", 7))
        self.algorithm = str(cfg.get("dedup.algorithm", "phash")).lower()
        if self.algorithm not in _ALGORITHMS:
            self.algorithm = "phash"
        self.hash_size = int(cfg.get("dedup.hash_size", 8))
        self.scope = str(cfg.get("dedup.scope", "video")).lower()
        self.window_seconds = float(cfg.get("dedup.window_seconds", 0) or 0)
        self._window_size = max(1, int(cfg.get("dedup.window_size", 240)))

        if self._window.maxlen != self._window_size:
            self._window = deque(self._window, maxlen=self._window_size)

    def start_video(self) -> None:
        """Called at the start of each video; clears the window if scoped."""
        self._video_key += 1
        if self.scope == "video":
            self._window.clear()

    def compute_hash(self, crop_bgr: np.ndarray):
        """Perceptual hash of a BGR crop, or None if it cannot be hashed."""
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            return None

        try:
            rgb = crop_bgr[:, :, ::-1]  # BGR -> RGB
            image = Image.fromarray(np.ascontiguousarray(rgb))
            if self.algorithm == "dhash":
                return imagehash.dhash(image, hash_size=self.hash_size)
            if self.algorithm == "ahash":
                return imagehash.average_hash(image, hash_size=self.hash_size)
            if self.algorithm == "whash":
                # whash needs a power-of-two hash size.
                size = 1 << max(2, int(self.hash_size) - 1).bit_length() - 1
                return imagehash.whash(image, hash_size=min(16, size))
            return imagehash.phash(image, hash_size=self.hash_size)
        except Exception as exc:
            log.debug("Hashing failed: %s", exc)
            return None

    def is_duplicate(self, image_hash, timestamp_s: float = 0.0) -> bool:
        """True if `image_hash` matches a recent crop, visually and in time."""
        if not self.enabled or image_hash is None:
            return False
        self._checked += 1

        # Newest first, so the time window can stop the scan early.
        for video_key, seen_ts, seen_hash in reversed(self._window):
            same_video = video_key == self._video_key
            if same_video and self.window_seconds > 0:
                # Entries are appended in ascending timestamp order within a
                # video, so once one falls outside the window every earlier
                # entry for this video does too.
                if (timestamp_s - seen_ts) > self.window_seconds:
                    if self.scope == "video":
                        break  # window holds this video only - nothing older left
                    continue   # global scope: earlier videos may still match
            try:
                if (image_hash - seen_hash) <= self.distance:
                    self._dropped += 1
                    return True
            except Exception:
                continue
        return False

    def remember(self, image_hash, timestamp_s: float = 0.0) -> None:
        if image_hash is not None:
            self._window.append((self._video_key, timestamp_s, image_hash))

    def check_and_remember(
        self, crop_bgr: np.ndarray, timestamp_s: float = 0.0
    ) -> tuple[bool, str | None]:
        """Combined path used by the pipeline. -> (is_duplicate, hash_string)"""
        image_hash = self.compute_hash(crop_bgr)
        if image_hash is None:
            return False, None
        if self.is_duplicate(image_hash, timestamp_s):
            return True, str(image_hash)
        self.remember(image_hash, timestamp_s)
        return False, str(image_hash)

    @property
    def stats(self) -> dict:
        return {
            "checked": self._checked,
            "dropped": self._dropped,
            "window": len(self._window),
            "window_size": self._window_size,
        }
