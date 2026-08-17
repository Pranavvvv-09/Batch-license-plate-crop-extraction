"""Background worker.

Owns the job queue and runs independently of the web UI - close the browser,
the batch keeps going. One video failing (dead link, geo-block, corrupt
stream, network drop) is caught, logged and retried a bounded number of times;
it can never take the batch down.
"""

from __future__ import annotations

import threading
import time
import traceback

from . import db
from .config import load as load_config
from .detector import DetectorError
from .frames import FrameReadError
from .logging_setup import get
from .pipeline import VideoProcessor
from .resolver import ResolveError
from .storage import PlateWriter

log = get("worker")

# Failures that will never succeed on retry.
FATAL_PATTERNS = (
    "private video", "video unavailable", "removed by the user",
    "account associated with this video has been terminated",
    "unsupported scheme", "members-only", "sign in to confirm",
    "is not available in your country", "does not exist",
)


class BatchWorker:
    """Drains the pending queue until stopped."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._detector = None
        self._writer: PlateWriter | None = None
        self._state = "idle"
        self._detector_error: str | None = None
        self._started_at: float | None = None
        self._idle_since: float | None = None

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        with self._lock:
            if self._threads and any(t.is_alive() for t in self._threads):
                return
            cfg = load_config(force=True)
            db.init()

            self._writer = PlateWriter(cfg)
            self._writer.sweep_temp_files()

            self._stop.clear()
            if db.get_state("paused", False):
                self._pause.set()
            else:
                self._pause.clear()

            concurrency = max(1, int(cfg.get("runtime.concurrency", 1)))
            self._started_at = time.time()
            self._state = "starting"
            self._threads = []
            for i in range(concurrency):
                thread = threading.Thread(
                    target=self._run, name=f"worker-{i + 1}", daemon=True
                )
                thread.start()
                self._threads.append(thread)
            log.info("Worker started with %d thread(s)", concurrency)

    def stop(self, timeout: float = 30.0) -> None:
        log.info("Stopping worker...")
        self._stop.set()
        self._pause.clear()
        for thread in self._threads:
            thread.join(timeout=timeout)
        # Anything caught mid-video goes back to pending with its checkpoint.
        reclaimed = db.reclaim_orphans()
        if reclaimed:
            log.info("Requeued %d in-flight video(s) for resume", reclaimed)
        self._state = "stopped"

    def pause(self) -> None:
        self._pause.set()
        db.set_state("paused", True)
        self._state = "paused"
        log.info("Worker paused by user")

    def resume(self) -> None:
        self._pause.clear()
        db.set_state("paused", False)
        self._state = "running"
        log.info("Worker resumed by user")

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    # ---------------------------------------------------------------- state

    def status(self) -> dict:
        writer = self._writer
        guard = writer.guard if writer else None
        detector_info = self._detector.info() if self._detector else None
        return {
            "state": self._state,
            "running": self.running,
            "paused": self.paused,
            "threads": sum(1 for t in self._threads if t.is_alive()),
            "started_at": self._started_at,
            "idle_since": self._idle_since,
            "detector": detector_info,
            "detector_error": self._detector_error,
            "disk_paused": bool(guard.paused) if guard else False,
            "free_gb": round(guard.free_gb, 2) if guard else None,
            "output_dir": str(writer.output_dir) if writer else None,
            "output_files": writer.count_files() if writer else 0,
        }

    # ------------------------------------------------------------- internals

    def _ensure_detector(self):
        """Load the model once, shared by all worker threads."""
        with self._lock:
            if self._detector is not None:
                return self._detector
            cfg = load_config()
            self._state = "loading-model"
            self._detector = None
            try:
                from .detector import PlateDetector

                self._detector = PlateDetector(cfg)
                self._detector_error = None
                db.log_event("info", f"Detector ready: {self._detector.info()}")
            except DetectorError as exc:
                self._detector_error = str(exc)
                self._state = "error"
                log.error("Detector unavailable: %s", exc)
                db.log_event("error", f"Detector unavailable: {exc}")
                raise
            except Exception as exc:  # noqa: BLE001 - surface any load failure
                self._detector_error = f"{type(exc).__name__}: {exc}"
                self._state = "error"
                log.error("Detector failed to load: %s", exc, exc_info=True)
                db.log_event("error", f"Detector failed to load: {exc}")
                raise
            return self._detector

    def _run(self) -> None:
        try:
            detector = self._ensure_detector()
        except Exception:
            # Nothing can be processed without a model; leave jobs pending so
            # a fixed install picks them up on restart.
            return

        cfg = load_config()
        writer = self._writer or PlateWriter(cfg)
        processor = VideoProcessor(detector, writer, cfg)
        self._state = "paused" if self.paused else "running"

        while not self._stop.is_set():
            if self._pause.is_set():
                self._state = "paused"
                time.sleep(1.0)
                continue

            # Re-read config between videos so dashboard edits take effect.
            cfg = load_config()

            # Disk guard: hold the whole worker rather than writing junk.
            if not writer.guard.refresh(cfg):
                self._state = "disk-full"
                db.set_state("disk_paused", True)
                time.sleep(15.0)
                continue
            if db.get_state("disk_paused", False):
                db.set_state("disk_paused", False)

            video = db.claim_next_video()
            if video is None:
                if self._state != "idle":
                    self._state = "idle"
                    self._idle_since = time.time()
                    log.info("Queue empty; waiting for new links")
                time.sleep(2.0)
                continue

            self._state = "running"
            self._idle_since = None
            self._process_one(processor, video, cfg)

        self._state = "stopped"

    def _process_one(self, processor: VideoProcessor, video: dict, cfg) -> None:
        video_id = int(video["id"])
        url = str(video["url"])
        attempts = int(video.get("attempts") or 1)
        max_retries = int(cfg.get("runtime.max_retries", 2))

        try:
            stats = processor.process(video, cfg, self._stop, self._pause)

            if self._stop.is_set():
                # Shutdown mid-video: leave it pending so it resumes.
                db.update_video(video_id, status=db.STATUS_PENDING)
                db.log_event(
                    "info",
                    f"Interrupted at {stats.last_timestamp:.0f}s; will resume",
                    video_id,
                )
                return

            db.update_video(
                video_id, status=db.STATUS_DONE, finished_at=time.time(), error=None
            )
            db.log_event(
                "info",
                f"Completed: {stats.saved} plates from {stats.frames} frames",
                video_id,
            )

        except (ResolveError, FrameReadError) as exc:
            self._handle_failure(video_id, url, exc, attempts, max_retries, cfg)
        except Exception as exc:  # noqa: BLE001 - the batch must never die here
            log.error("[v%d] Unexpected error: %s\n%s", video_id, exc, traceback.format_exc())
            self._handle_failure(video_id, url, exc, attempts, max_retries, cfg)

    def _handle_failure(
        self, video_id: int, url: str, exc: Exception, attempts: int, max_retries: int, cfg
    ) -> None:
        message = f"{type(exc).__name__}: {exc}"[:1000]
        permanent = any(p in message.lower() for p in FATAL_PATTERNS)
        can_retry = (not permanent) and attempts <= max_retries

        if can_retry:
            backoff = float(cfg.get("runtime.retry_backoff_seconds", 20))
            log.warning(
                "[v%d] Failed (attempt %d/%d), retrying in %.0fs: %s",
                video_id, attempts, max_retries + 1, backoff, message,
            )
            db.update_video(video_id, status=db.STATUS_PENDING, error=message)
            db.log_event("warning", f"Retry {attempts}/{max_retries + 1}: {message}", video_id)
            # Sleep interruptibly so a stop request is still fast.
            self._stop.wait(timeout=backoff)
        else:
            log.error("[v%d] Giving up on %s: %s", video_id, url[:80], message)
            db.update_video(
                video_id, status=db.STATUS_ERROR, error=message, finished_at=time.time()
            )
            db.log_event("error", f"Failed permanently: {message}", video_id)


_worker: BatchWorker | None = None
_worker_lock = threading.Lock()


def get_worker() -> BatchWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = BatchWorker()
        return _worker
