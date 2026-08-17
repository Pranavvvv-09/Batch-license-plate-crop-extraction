"""Per-video processing pipeline.

For one video, in order:

  1. resolve stream            (resolver)
  2. sample frames             (frames)
  3. detect plates             (detector)
  4. confidence filter         (detector conf threshold)
  5. size / aspect filter      (quality.passes_geometry)
  6. tight crop                (quality.crop)
  7. blur / contrast gate      (quality.assess)
  8. perceptual-hash dedup     (dedup)
  9. atomic JPEG save          (storage)

No OCR anywhere - the deliverable is images, not text.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from . import db
from .config import Config, load as load_config
from .dedup import Deduplicator
from .detector import PlateDetector
from .frames import Frame, FrameReadError, iter_frames
from .logging_setup import get
from .quality import assess, crop, passes_geometry
from .resolver import ResolveError, resolve
from .storage import PlateWriter

log = get("pipeline")


@dataclass
class VideoStats:
    frames: int = 0
    detections: int = 0
    saved: int = 0
    rejected_size: int = 0
    rejected_blur: int = 0
    rejected_dupe: int = 0
    rejected_write: int = 0
    last_timestamp: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        key = reason.split("(")[0]
        self.reasons[key] = self.reasons.get(key, 0) + 1


class VideoProcessor:
    """Processes one video at a time using a shared detector and writer."""

    def __init__(self, detector: PlateDetector, writer: PlateWriter, cfg: Config):
        self.detector = detector
        self.writer = writer
        self.dedup = Deduplicator(cfg)

    def process(
        self,
        video: dict,
        cfg: Config,
        stop_event: threading.Event,
        pause_event: threading.Event | None = None,
    ) -> VideoStats:
        """Run one video end to end. Raises on unrecoverable ingest failure."""
        video_id = int(video["id"])
        url = str(video["url"])
        resume_at = float(video.get("position_s") or 0.0)

        source = resolve(url, cfg)  # ResolveError propagates to the worker
        title = source.title or url
        db.update_video(
            video_id, title=title[:200], duration_s=source.duration_s,
            heartbeat_at=time.time(),
        )
        if resume_at > 0:
            log.info("[v%d] Resuming '%s' at %.0fs", video_id, title[:60], resume_at)
        else:
            log.info("[v%d] Starting '%s'", video_id, title[:60])

        self.dedup.configure(cfg)
        self.dedup.start_video()

        stats = VideoStats(last_timestamp=resume_at)
        batch_size = max(1, int(cfg.get("detector.batch_size", 8)))
        checkpoint_every = max(1, int(cfg.get("runtime.checkpoint_every_n_frames", 30)))
        max_plates = int(cfg.get("runtime.max_plates_per_video", 0) or 0)
        timeout = float(cfg.get("runtime.per_video_timeout", 0) or 0)
        started = time.time()

        batch: list[Frame] = []
        frames_since_checkpoint = 0

        def flush() -> None:
            nonlocal batch, frames_since_checkpoint
            if not batch:
                return
            results = self.detector.detect_batch([f.image for f in batch], cfg)
            for frame, detections in zip(batch, results):
                self._handle_frame(video_id, title, frame, detections, cfg, stats)
            stats.frames += len(batch)
            stats.last_timestamp = batch[-1].timestamp_s
            frames_since_checkpoint += len(batch)
            batch = []

            if frames_since_checkpoint >= checkpoint_every:
                self._checkpoint(video_id, stats)
                frames_since_checkpoint = 0

        try:
            for frame in iter_frames(source, cfg, start_at_s=resume_at, stop_event=stop_event):
                if stop_event.is_set():
                    break

                # User pause, or the disk guard tripped. Hold here rather than
                # burning GPU and bandwidth on frames that cannot be saved.
                # The checkpoint is written first, so a kill while paused
                # still resumes from the right place.
                held_for = self._hold_reason(pause_event, cfg)
                if held_for:
                    flush()
                    self._checkpoint(video_id, stats)
                    log.info(
                        "[v%d] Holding at %.0fs (%s)",
                        video_id, stats.last_timestamp, held_for,
                    )
                    while not stop_event.is_set() and self._hold_reason(pause_event, cfg):
                        stop_event.wait(timeout=2.0)
                    if stop_event.is_set():
                        break
                    log.info("[v%d] Resuming at %.0fs", video_id, stats.last_timestamp)

                batch.append(frame)
                if len(batch) >= batch_size:
                    flush()

                if max_plates and stats.saved >= max_plates:
                    log.info("[v%d] Hit max_plates_per_video (%d)", video_id, max_plates)
                    break
                if timeout and (time.time() - started) > timeout:
                    log.warning("[v%d] Hit per_video_timeout (%.0fs)", video_id, timeout)
                    break

            flush()
        except FrameReadError:
            flush()
            self._checkpoint(video_id, stats)
            raise
        finally:
            self._checkpoint(video_id, stats)

        log.info(
            "[v%d] Done: %d frames, %d detections, %d saved "
            "(size %d, blur %d, dupe %d)",
            video_id, stats.frames, stats.detections, stats.saved,
            stats.rejected_size, stats.rejected_blur, stats.rejected_dupe,
        )
        return stats

    # ------------------------------------------------------------ internals

    def _hold_reason(self, pause_event: threading.Event | None, cfg: Config) -> str:
        """Why processing should hold, or '' to keep going."""
        if pause_event is not None and pause_event.is_set():
            return "paused by user"
        if not self.writer.guard.refresh(cfg):
            return f"disk guard: {self.writer.guard.free_gb:.2f} GB free"
        return ""

    def _handle_frame(
        self,
        video_id: int,
        title: str,
        frame: Frame,
        detections: list,
        cfg: Config,
        stats: VideoStats,
    ) -> None:
        if not detections:
            return

        height, width = frame.image.shape[:2]
        stats.detections += len(detections)
        # Best-first, so a per-video plate cap keeps the strongest crops.
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        for seq, det in enumerate(detections):
            ok, reason = passes_geometry(det, width, height, cfg)
            if not ok:
                stats.rejected_size += 1
                stats.note(reason)
                continue

            region = crop(frame.image, det, cfg)
            if region is None:
                stats.rejected_size += 1
                stats.note("crop_failed")
                continue

            report = assess(region, cfg)
            if not report.accepted:
                stats.rejected_blur += 1
                stats.note(report.reason)
                continue

            is_dupe, phash = self.dedup.check_and_remember(region, frame.timestamp_s)
            if is_dupe:
                stats.rejected_dupe += 1
                stats.note("duplicate")
                continue

            if not self.writer.guard.refresh(cfg):
                stats.note("disk_paused")
                return

            filename = self.writer.build_name(
                video_id, title, frame.index, frame.timestamp_s, seq
            )
            result = self.writer.save(region, filename, cfg)
            if not result.ok:
                stats.rejected_write += 1
                stats.note(result.reason)
                continue

            crop_h, crop_w = region.shape[:2]
            db.record_plate(
                video_id=video_id,
                filename=filename,
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                confidence=det.confidence,
                box=det.as_tuple(),
                width=crop_w,
                height=crop_h,
                blur_score=report.blur_score,
                phash=phash,
                nbytes=result.nbytes,
                storage_url=result.storage_url,
            )

            self.writer.append_manifest({
                "file": filename,
                "video_id": video_id,
                "source_title": title[:200],
                "timestamp_s": round(frame.timestamp_s, 3),
                "frame_index": frame.index,
                "confidence": round(det.confidence, 4),
                "box_xyxy": list(det.as_tuple()),
                "crop_wh": [crop_w, crop_h],
                "blur_score": round(report.blur_score, 2),
                "phash": phash,
                "bytes": result.nbytes,
                "saved_at": round(time.time(), 3),
            })
            stats.saved += 1

    def _checkpoint(self, video_id: int, stats: VideoStats) -> None:
        """Persist progress so a kill/restart resumes instead of restarting."""
        try:
            db.update_video(
                video_id,
                frames_processed=stats.frames,
                plates_saved=stats.saved,
                detections_seen=stats.detections,
                rejected_size=stats.rejected_size,
                rejected_blur=stats.rejected_blur,
                rejected_dupe=stats.rejected_dupe,
                position_s=stats.last_timestamp,
                heartbeat_at=time.time(),
            )
        except Exception as exc:
            log.warning("[v%d] Checkpoint failed: %s", video_id, exc)


def build_processor(cfg: Config | None = None) -> VideoProcessor:
    cfg = cfg or load_config()
    return VideoProcessor(PlateDetector(cfg), PlateWriter(cfg), cfg)


__all__ = [
    "VideoProcessor", "VideoStats", "build_processor",
    "ResolveError", "FrameReadError",
]
