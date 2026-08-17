"""Frame sampling.

Frames are pulled from the network one at a time and discarded after use; the
source video is never written to disk.

Primary path: ffmpeg with an `fps=N` filter, emitting raw BGR frames on a
pipe. Only the sampled frames cross the pipe, so bandwidth and memory stay
flat regardless of source length.

Fallback path: OpenCV VideoCapture with grab()-skipping, used when ffmpeg is
unavailable or the frame geometry cannot be determined up front.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import threading
from typing import Iterator, NamedTuple

import numpy as np

from .config import Config
from .logging_setup import get
from .resolver import StreamSource

log = get("frames")


class Frame(NamedTuple):
    index: int          # sampled-frame counter (0-based, within this pass)
    timestamp_s: float  # approximate position in the source video
    image: np.ndarray   # BGR uint8


class FrameReadError(RuntimeError):
    """Raised when a stream cannot be opened or dies mid-read."""


@functools.lru_cache(maxsize=1)
def ffmpeg_binary() -> str | None:
    """System ffmpeg if present, else the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _target_size(
    src_w: int, src_h: int, max_width: int
) -> tuple[int, int]:
    """Even-dimensioned output size after the optional downscale."""
    if max_width and src_w > max_width:
        scale = max_width / float(src_w)
        out_w = int(round(src_w * scale))
        out_h = int(round(src_h * scale))
    else:
        out_w, out_h = src_w, src_h
    return max(2, out_w - (out_w % 2)), max(2, out_h - (out_h % 2))


class _StderrDrain(threading.Thread):
    """Keeps ffmpeg's stderr pipe from filling and deadlocking the decoder."""

    def __init__(self, stream, keep: int = 40):
        super().__init__(daemon=True, name="ffmpeg-stderr")
        self._stream = stream
        self._keep = keep
        self.lines: list[str] = []

    def run(self) -> None:
        try:
            for raw in iter(self._stream.readline, b""):
                text = raw.decode("utf-8", "replace").strip()
                if text:
                    self.lines.append(text)
                    if len(self.lines) > self._keep:
                        del self.lines[0]
        except Exception:
            pass

    def tail(self, n: int = 6) -> str:
        return " | ".join(self.lines[-n:])


def iter_frames(
    source: StreamSource,
    cfg: Config,
    start_at_s: float = 0.0,
    stop_event: threading.Event | None = None,
) -> Iterator[Frame]:
    """Yield sampled frames from `source`, resuming at `start_at_s`."""
    if source.kind == "image":
        yield from _iter_image(source, cfg)
        return

    fps = float(cfg.get("sampling.fps", 1.0))
    if fps <= 0:
        fps = 1.0

    use_ffmpeg = bool(cfg.get("ingest.prefer_ffmpeg", True)) and ffmpeg_binary()
    if use_ffmpeg:
        try:
            yield from _iter_ffmpeg(source, cfg, fps, start_at_s, stop_event)
            return
        except FrameReadError:
            raise
        except Exception as exc:
            log.warning("ffmpeg path failed (%s); falling back to OpenCV", exc)

    if source.pipe_cmd and not source.url:
        raise FrameReadError(
            "This link needs the ffmpeg pipe, which is unavailable. "
            "Install ffmpeg or `pip install imageio-ffmpeg`."
        )
    yield from _iter_opencv(source, cfg, fps, start_at_s, stop_event)


# --------------------------------------------------------------- ffmpeg path


def _resolve_geometry(source: StreamSource, cfg: Config) -> tuple[int, int]:
    """Frame size must be known up front to size the rawvideo pipe reads."""
    if source.width and source.height:
        return int(source.width), int(source.height)

    from .resolver import probe_dimensions

    probed = probe_dimensions(source, cfg)
    if probed:
        # Cache on the source so a retry of the same video skips the probe.
        source.width, source.height = probed
        return probed
    raise FrameReadError("could not determine video dimensions")


def _iter_ffmpeg(
    source: StreamSource,
    cfg: Config,
    fps: float,
    start_at_s: float,
    stop_event: threading.Event | None,
) -> Iterator[Frame]:
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not available")

    src_w, src_h = _resolve_geometry(source, cfg)
    out_w, out_h = _target_size(src_w, src_h, int(cfg.get("sampling.max_frame_width", 0) or 0))
    frame_bytes = out_w * out_h * 3
    read_timeout = float(cfg.get("ingest.read_timeout", 90))

    cmd: list[str] = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-fflags", "+discardcorrupt",
    ]

    puller: subprocess.Popen | None = None
    if source.url:
        if source.headers:
            blob = "".join(f"{k}: {v}\r\n" for k, v in source.headers.items())
            cmd += ["-headers", blob]
        cmd += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "10",
            "-rw_timeout", str(int(read_timeout * 1_000_000)),
        ]
        # Input-side seek: fast, no decoding of the skipped span.
        if start_at_s > 0:
            cmd += ["-ss", f"{start_at_s:.3f}"]
        cmd += ["-i", source.url]
    else:
        cmd += ["-i", "pipe:0"]
        # Output-side seek on a pipe: still decodes the skipped span, but skips
        # detection on it, which is where the real cost is.
        if start_at_s > 0:
            cmd += ["-ss", f"{start_at_s:.3f}"]

    cmd += [
        "-an", "-sn", "-dn",
        "-vf", f"fps={fps},scale={out_w}:{out_h}:flags=bicubic",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    if source.pipe_cmd:
        puller = subprocess.Popen(
            source.pipe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1024 * 1024,
        )

    proc = subprocess.Popen(
        cmd,
        stdin=(puller.stdout if puller else subprocess.DEVNULL),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes,
    )
    if puller and puller.stdout:
        # ffmpeg owns the read end now; closing ours makes SIGPIPE work.
        puller.stdout.close()

    drain = _StderrDrain(proc.stderr)
    drain.start()

    log.info(
        "ffmpeg stream open: %dx%d @ %.2f fps%s",
        out_w, out_h, fps, f", resuming at {start_at_s:.0f}s" if start_at_s > 0 else "",
    )

    index = 0
    produced = 0
    try:
        assert proc.stdout is not None
        while True:
            if stop_event is not None and stop_event.is_set():
                log.info("Stop requested; closing stream")
                break

            chunk = _read_exact(proc.stdout, frame_bytes)
            if chunk is None:
                break  # clean end of stream
            if len(chunk) < frame_bytes:
                break  # truncated tail frame

            # Copy out of the pipe buffer: frombuffer is read-only, and the
            # detector/cropper both want a writable array.
            image = np.frombuffer(chunk, dtype=np.uint8).reshape(out_h, out_w, 3).copy()
            timestamp = start_at_s + (index / fps)
            index += 1
            produced += 1
            yield Frame(index - 1, timestamp, image)

        code = proc.poll()
        if produced == 0:
            raise FrameReadError(f"no frames decoded: {drain.tail() or f'exit code {code}'}")
        if code not in (0, None):
            # Partial read. The caller has a checkpoint and will resume.
            log.warning("ffmpeg exited %s after %d frames: %s", code, produced, drain.tail())
    finally:
        for handle in (proc.stdout,):
            try:
                if handle:
                    handle.close()
            except Exception:
                pass
        for child in (proc, puller):
            if child is None:
                continue
            if child.poll() is None:
                try:
                    child.terminate()
                    child.wait(timeout=5)
                except Exception:
                    try:
                        child.kill()
                    except Exception:
                        pass


def _read_exact(stream, size: int) -> bytes | None:
    """Read exactly `size` bytes. None on clean EOF, short bytes on truncation."""
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)
        buf.extend(chunk)
    return bytes(buf)


# -------------------------------------------------------------- OpenCV path


def _iter_opencv(
    source: StreamSource,
    cfg: Config,
    fps: float,
    start_at_s: float,
    stop_event: threading.Event | None,
) -> Iterator[Frame]:
    import cv2

    url = source.url
    if not url:
        raise FrameReadError("OpenCV fallback needs a direct URL")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise FrameReadError("OpenCV could not open the stream")

    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if native_fps <= 0 or native_fps > 240:
            native_fps = 30.0
        step = max(1, int(round(native_fps / fps)))
        max_width = int(cfg.get("sampling.max_frame_width", 0) or 0)

        if start_at_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_at_s * 1000.0)

        log.info(
            "OpenCV stream open: native %.2f fps, sampling every %d frames", native_fps, step
        )

        index = 0
        raw_index = 0
        produced = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            ok, image = cap.read()
            if not ok or image is None:
                break
            if raw_index % step == 0:
                if max_width and image.shape[1] > max_width:
                    scale = max_width / float(image.shape[1])
                    image = cv2.resize(
                        image,
                        (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                timestamp = start_at_s + (raw_index / native_fps)
                yield Frame(index, timestamp, image)
                index += 1
                produced += 1
            raw_index += 1

        if produced == 0:
            raise FrameReadError("OpenCV decoded no frames")
    finally:
        try:
            cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------- Image path


def _iter_image(source: StreamSource, cfg: Config) -> Iterator[Frame]:
    """Yield a single frame from an image source (Google Drive or direct image link)."""
    import cv2
    from .gdrive import is_gdrive_url, fetch_gdrive_image

    url = source.url or ""
    image: np.ndarray | None = None

    if is_gdrive_url(url):
        try:
            image, title = fetch_gdrive_image(url)
            if title and not source.title:
                source.title = title
        except Exception as exc:
            raise FrameReadError(f"Could not fetch Google Drive image: {exc}") from exc
    else:
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "sang_det/1.0"})
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            log.warning("Direct image fetch failed (%s); trying OpenCV imread", exc)
            image = cv2.imread(url)

    if image is None:
        raise FrameReadError(f"Could not decode image from source: {url}")

    max_width = int(cfg.get("sampling.max_frame_width", 0) or 0)
    if max_width and image.shape[1] > max_width:
        scale = max_width / float(image.shape[1])
        new_w = max(2, int(round(image.shape[1] * scale)))
        new_h = max(2, int(round(image.shape[0] * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    yield Frame(0, 0.0, image)
