"""Turn a submitted link into something ffmpeg/OpenCV can read.

Two shapes come back:

  * a direct HTTP(S) media URL plus any headers required to fetch it, or
  * a `yt-dlp -o -` command whose stdout is piped into the decoder.

Nothing is ever written to disk. The full video is never stored - frames are
pulled off the wire, used, and dropped.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .gdrive import is_gdrive_url, parse_gdrive_url, get_gdrive_download_url, fetch_gdrive_image
from .logging_setup import get

log = get("resolver")

DIRECT_MEDIA_RE = re.compile(
    r"\.(mp4|m4v|mov|mkv|avi|webm|flv|wmv|mpg|mpeg|ts|m3u8|mpd)(\?|$)", re.IGNORECASE
)
DIRECT_IMAGE_RE = re.compile(
    r"\.(jpg|jpeg|png|webp|bmp|tiff|tif|jfif|heic)(\?|$)", re.IGNORECASE
)
# Hosts we always hand to yt-dlp even without a media extension.
EXTRACTOR_HINTS = (
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv",
    "facebook.com", "bilibili.com", "rumble.com", "odysee.com", "streamable.com",
)


class ResolveError(RuntimeError):
    """Raised when a link cannot be turned into a readable stream."""


@dataclass
class StreamSource:
    """A resolved, readable video stream or image source."""

    url: str | None = None                        # direct URL for ffmpeg -i / image fetch
    pipe_cmd: list[str] | None = None             # or a command to pipe from
    headers: dict[str, str] = field(default_factory=dict)
    title: str = ""
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    is_live: bool = False
    kind: str = "direct"                          # direct | extracted | piped | image

    @property
    def seekable(self) -> bool:
        """Only a real URL supports fast input-side seeking for resume."""
        return self.url is not None and not self.is_live and self.kind != "image"


def _looks_direct(url: str) -> bool:
    lowered = url.lower()
    if is_gdrive_url(lowered) or any(hint in lowered for hint in EXTRACTOR_HINTS):
        return False
    return bool(DIRECT_MEDIA_RE.search(lowered))


def _looks_image(url: str) -> bool:
    lowered = url.lower()
    if is_gdrive_url(lowered) or any(hint in lowered for hint in EXTRACTOR_HINTS):
        return False
    return bool(DIRECT_IMAGE_RE.search(lowered))


def _ytdlp_binary() -> list[str]:
    """Prefer the console script; fall back to `python -m yt_dlp`."""
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def _format_selector(max_height: int) -> str:
    """Video-only, H.264-preferred, height-capped. No audio - we never use it."""
    h = int(max_height)
    return (
        f"bestvideo[height<={h}][vcodec^=avc1]/"
        f"bestvideo[height<={h}]/"
        f"best[height<={h}]/"
        "bestvideo/best"
    )


def _preflight(url: str, cfg: Config) -> tuple[bool, str]:
    """Fast reachability probe for a direct URL. -> (ok, reason_if_not)"""
    import urllib.error
    import urllib.request

    timeout = min(20.0, float(cfg.get("ingest.socket_timeout", 30)))
    request = urllib.request.Request(
        url,
        method="GET",
        # Ask for one byte: works on servers that reject HEAD, and avoids
        # pulling the file just to prove it exists.
        headers={"Range": "bytes=0-0", "User-Agent": "sang_det/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = getattr(response, "status", 200)
            if code >= 400:
                return False, f"HTTP {code} fetching video"
            return True, ""
    except urllib.error.HTTPError as exc:
        if exc.code in (405, 416, 501):  # method/range unsupported, still alive
            return True, ""
        return False, f"HTTP {exc.code} fetching video"
    except urllib.error.URLError as exc:
        return False, f"cannot reach host: {exc.reason}"
    except Exception as exc:  # socket timeouts, malformed URLs, TLS failures
        return False, f"cannot reach host: {exc}"


def _probe_direct(url: str) -> tuple[int | None, int | None, float | None, float | None]:
    """Read dimensions/fps/duration from a direct URL using OpenCV.

    ffprobe is not assumed to exist (the bundled imageio-ffmpeg ships ffmpeg
    only), so OpenCV opens the stream just long enough to read its header.
    """
    try:
        import cv2
    except ImportError:
        return None, None, None, None

    cap = None
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return None, None, None, None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
        duration = (frames / fps) if (fps and frames > 0) else None

        # Some containers/streams do not populate the size properties until a
        # frame has actually been decoded. Pay for one frame to find out.
        if not width or not height:
            ok, image = cap.read()
            if ok and image is not None:
                height, width = image.shape[:2]

        return width, height, fps, duration
    except Exception:
        return None, None, None, None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def resolve(url: str, cfg: Config) -> StreamSource:
    """Resolve a submitted link. Raises ResolveError if nothing works."""
    url = url.strip()
    if not url:
        raise ResolveError("empty URL")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ResolveError(f"unsupported scheme: {url[:60]}")

    if is_gdrive_url(url):
        kind, target_id = parse_gdrive_url(url)
        if kind == "folder":
            raise ResolveError(
                "Google Drive folder link submitted directly. Please scan or queue individual files."
            )
        file_id = target_id or url
        image_err: Exception | None = None
        try:
            image, title = fetch_gdrive_image(url)
            h, w = image.shape[:2]
            log.info("Google Drive image: '%s' (%dx%d)", title, w, h)
            return StreamSource(
                url=url, title=title[:120], width=w, height=h,
                fps=1.0, duration_s=1.0, kind="image",
            )
        except Exception as exc:
            image_err = exc
            log.debug("Google Drive image decode failed (%s); checking if video", exc)

        # If not an image, check if it's a reachable Google Drive video stream
        direct_download = get_gdrive_download_url(file_id)
        reachable, _ = _preflight(direct_download, cfg)
        if reachable:
            width, height, fps, duration = _probe_direct(direct_download)
            if width and height:
                log.info("Google Drive video: %s (%sx%s, %.0fs)", file_id, width, height, duration or 0)
                return StreamSource(
                    url=direct_download, title=f"gdrive_{file_id}", width=width,
                    height=height, fps=fps, duration_s=duration, kind="direct",
                )

        raise ResolveError(
            f"Could not load Google Drive file ({file_id}): {image_err or 'File not found or permission denied'}"
        )

    if _looks_image(url):
        reachable, detail = _preflight(url, cfg)
        if not reachable:
            raise ResolveError(detail)
        import cv2
        cap = cv2.VideoCapture(url)
        w, h = None, None
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            cap.release()
        stem = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        stem = DIRECT_IMAGE_RE.sub("", stem) or "image"
        return StreamSource(
            url=url, title=stem[:120], width=w, height=h,
            fps=1.0, duration_s=1.0, kind="image",
        )

    if _looks_direct(url):
        # Cheap reachability check first. Without it a dead host burns ~40s per
        # attempt in the OpenCV and ffmpeg probes and reports the misleading
        # "could not determine video dimensions" instead of the real cause.
        reachable, detail = _preflight(url, cfg)
        if not reachable:
            raise ResolveError(detail)

        width, height, fps, duration = _probe_direct(url)
        log.info("Direct media URL (%sx%s, %.0fs)", width, height, duration or 0)
        # Filename without query string or extension - it becomes the title,
        # and the title becomes part of every output filename.
        stem = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        stem = DIRECT_MEDIA_RE.sub("", stem) or "video"
        return StreamSource(
            url=url, title=stem[:120], width=width,
            height=height, fps=fps, duration_s=duration, kind="direct",
        )

    return _resolve_with_ytdlp(url, cfg)


def _resolve_with_ytdlp(url: str, cfg: Config) -> StreamSource:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise ResolveError("yt-dlp is not installed") from exc

    max_height = int(cfg.get("ingest.max_height", 1080))
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": _format_selector(max_height),
        "socket_timeout": int(cfg.get("ingest.socket_timeout", 30)),
        "retries": 3,
        "extractor_retries": 2,
        "nocheckcertificate": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise ResolveError(f"yt-dlp could not resolve link: {exc}") from exc

    if info is None:
        raise ResolveError("yt-dlp returned no metadata")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise ResolveError("playlist contained no playable entries")
        info = entries[0]

    chosen = _pick_format(info, max_height)
    title = str(info.get("title") or url)[:200]
    duration = info.get("duration")
    is_live = bool(info.get("is_live"))

    source = StreamSource(
        title=title,
        duration_s=float(duration) if duration else None,
        is_live=is_live,
        kind="extracted",
    )

    if chosen:
        proto = str(chosen.get("protocol") or "")
        direct_url = chosen.get("url")
        source.width = chosen.get("width")
        source.height = chosen.get("height")
        source.fps = chosen.get("fps")
        # ffmpeg reads https/m3u8 natively. Fragmented DASH and anything
        # exotic goes through the yt-dlp pipe instead.
        if direct_url and proto.startswith(("http", "m3u8")):
            source.url = direct_url
            source.headers = _clean_headers(
                {**(info.get("http_headers") or {}), **(chosen.get("http_headers") or {})}
            )
            log.info("Resolved '%s' -> %s (%sp, %s)", title[:60],
                     chosen.get("format_id"), source.height, proto)
            return source

    source.kind = "piped"
    source.pipe_cmd = [
        *_ytdlp_binary(),
        "--quiet", "--no-warnings", "--no-playlist",
        "--no-part", "--no-continue",
        "--socket-timeout", str(int(cfg.get("ingest.socket_timeout", 30))),
        "--retries", "3",
        "-f", _format_selector(max_height),
        "-o", "-",
        url,
    ]
    log.info("Resolved '%s' -> yt-dlp stdout pipe", title[:60])
    return source


def _pick_format(info: dict[str, Any], max_height: int) -> dict[str, Any] | None:
    """Best height-capped video track; video-only preferred over muxed."""
    if info.get("url") and not info.get("formats"):
        return {
            "url": info["url"],
            "protocol": info.get("protocol", "https"),
            "width": info.get("width"),
            "height": info.get("height"),
            "fps": info.get("fps"),
            "format_id": info.get("format_id", "direct"),
            "http_headers": info.get("http_headers"),
        }

    formats = [f for f in (info.get("formats") or []) if f.get("url")]
    formats = [f for f in formats if f.get("vcodec") not in (None, "none")]
    if not formats:
        return None

    def score(f: dict[str, Any]) -> tuple:
        height = f.get("height") or 0
        fits = height <= max_height
        proto = str(f.get("protocol") or "")
        return (
            1 if fits else 0,                                  # prefer within cap
            height if fits else -height,                       # then biggest that fits
            1 if f.get("acodec") in (None, "none") else 0,     # video-only is lighter
            1 if proto.startswith("http") else 0,              # plain HTTP is simplest
            1 if str(f.get("vcodec", "")).startswith("avc1") else 0,
            f.get("tbr") or 0,
        )

    return max(formats, key=score)


def _clean_headers(headers: dict[str, Any]) -> dict[str, str]:
    """Drop headers ffmpeg sets itself; keep auth/UA/referer."""
    drop = {"accept-encoding", "range", "connection", "host"}
    return {
        str(k): str(v)
        for k, v in (headers or {}).items()
        if k and v and str(k).lower() not in drop
    }


def probe_dimensions(source: StreamSource, cfg: Config) -> tuple[int, int] | None:
    """Ask ffmpeg itself for the decoded frame size.

    Works for both direct URLs and yt-dlp pipes, and is the authority when
    container metadata is missing or lying. ffprobe is not required - ffmpeg
    reports the stream geometry on stderr while decoding a single frame.
    """
    from .frames import ffmpeg_binary  # local import avoids a cycle

    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        return None

    cmd = [ffmpeg, "-hide_banner", "-nostdin"]
    puller = None
    probe = None
    try:
        if source.url:
            if source.headers:
                blob = "".join(f"{k}: {v}\r\n" for k, v in source.headers.items())
                cmd += ["-headers", blob]
            cmd += ["-i", source.url]
            stdin = subprocess.DEVNULL
        elif source.pipe_cmd:
            puller = subprocess.Popen(
                source.pipe_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            cmd += ["-i", "pipe:0"]
            stdin = puller.stdout
        else:
            return None

        cmd += ["-frames:v", "1", "-f", "null", "-"]
        probe = subprocess.Popen(
            cmd, stdin=stdin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if puller and puller.stdout:
            puller.stdout.close()

        _, stderr = probe.communicate(timeout=float(cfg.get("ingest.read_timeout", 90)) + 30)
        match = re.search(rb"Stream .*?Video:.*?[, ](\d{2,5})x(\d{2,5})", stderr or b"")
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            log.info("ffmpeg reports stream geometry: %dx%d", width, height)
            return width, height
        log.warning("ffmpeg did not report a video stream size")
    except Exception as exc:
        log.debug("Dimension probe failed: %s", exc)
    finally:
        for proc in (probe, puller):
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
    return None
