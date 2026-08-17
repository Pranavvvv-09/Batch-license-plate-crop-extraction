"""Configuration loading.

config.yaml is the single source of truth for every tunable. It is read on
demand (cheaply, with an mtime cache) so the worker picks up dashboard edits
between videos without a restart.
"""

from __future__ import annotations

import copy
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("SANG_DET_CONFIG", ROOT / "config.yaml"))


# Mirrors config.yaml. Used to fill gaps if the file is partial or corrupt,
# so a bad edit degrades to defaults instead of taking the run down.
DEFAULTS: dict[str, Any] = {
    "sampling": {"fps": 1.0, "max_frame_width": 1920},
    "detector": {
        "model_path": "",
        "imgsz": 960,
        "confidence": 0.35,
        "iou": 0.45,
        "device": "auto",
        "batch_size": 8,
        "half": True,
        "max_detections": 20,
    },
    "filters": {
        "min_box_width": 60,
        "min_box_height": 20,
        "min_aspect_ratio": 1.2,
        "max_aspect_ratio": 8.0,
        "max_area_fraction": 0.25,
        "crop_padding_px": 3,
    },
    "quality": {
        "enabled": True,
        "min_laplacian_variance": 45.0,
        "min_contrast_stddev": 18.0,
        "max_uniform_fraction": 0.92,
        "min_crop_pixels": 1200,
    },
    "dedup": {
        "enabled": True,
        # hash_distance is calibrated against hash_size; see config.yaml.
        "hash_distance": 7,
        "window_size": 240,
        "window_seconds": 20.0,
        "scope": "video",
        "algorithm": "phash",
        "hash_size": 8,
    },
    "output": {
        "dir": "data/output",
        "jpeg_quality": 92,
        "min_save_width": 0,
        "manifest": "data/manifest.jsonl",
    },
    "storage": {
        "min_free_gb": 5.0,
        "check_every_n_saves": 50,
        "check_every_n_seconds": 60,
    },
    "runtime": {
        "concurrency": 1,
        "max_retries": 2,
        "retry_backoff_seconds": 20,
        "per_video_timeout": 0,
        "checkpoint_every_n_frames": 30,
        "max_plates_per_video": 0,
    },
    "ingest": {
        "max_height": 1080,
        "read_timeout": 90,
        "socket_timeout": 30,
        "prefer_ffmpeg": True,
    },
    "server": {"host": "127.0.0.1", "port": 8000, "autostart_worker": True},
    "logging": {
        "level": "INFO",
        "file": "data/logs/sang_det.log",
        "max_bytes": 10485760,
        "backup_count": 5,
    },
    "labeler": {
        "enabled": True,
        "api_key": "",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "fallback_model": "meta/llama-3.2-11b-vision-instruct",
        "prompt": "Extract and output only the license plate alphanumeric text from this image. Output uppercase text with standard spacing. If unreadable, output UNREADABLE. Do not include any explanations or other text.",
        "timeout": 60.0,
        "reasoning_budget": 128,
        "temperature": 0.1,
        "max_tokens": 128,
        "csv_path": "data/labels.csv",
        "rate_limit_rps": 1.5,
        "min_delay_s": 0.0,
        "max_retries": 5,
        "retry_base_delay_s": 2.0,
        "retry_max_delay_s": 45.0,
        "batch_chunk_size": 50,
    },
    "supabase": {
        "url": "",
        "key": "",
        "bucket": "plates",
        "db_url": "",
    },
}

# Tunables the dashboard is allowed to write. Anything outside this set is
# file-only, so a browser cannot repoint the output directory mid-run.
EDITABLE_KEYS = {
    "sampling.fps",
    "sampling.max_frame_width",
    "detector.imgsz",
    "detector.confidence",
    "detector.iou",
    "detector.batch_size",
    "filters.min_box_width",
    "filters.min_box_height",
    "filters.min_aspect_ratio",
    "filters.max_aspect_ratio",
    "filters.max_area_fraction",
    "filters.crop_padding_px",
    "quality.enabled",
    "quality.min_laplacian_variance",
    "quality.min_contrast_stddev",
    "dedup.enabled",
    "dedup.hash_distance",
    "dedup.window_size",
    "dedup.window_seconds",
    "dedup.scope",
    "output.jpeg_quality",
    "storage.min_free_gb",
    "runtime.concurrency",
    "runtime.max_retries",
    "runtime.max_plates_per_video",
    "ingest.max_height",
    "labeler.enabled",
    "labeler.api_key",
    "labeler.base_url",
    "labeler.model",
    "labeler.fallback_model",
    "labeler.prompt",
    "labeler.reasoning_budget",
    "labeler.timeout",
    "labeler.temperature",
    "labeler.max_tokens",
    "labeler.csv_path",
    "labeler.rate_limit_rps",
    "labeler.min_delay_s",
    "labeler.max_retries",
    "labeler.retry_base_delay_s",
    "labeler.retry_max_delay_s",
    "labeler.batch_chunk_size",
    "supabase.url",
    "supabase.key",
    "supabase.bucket",
    "supabase.db_url",
}

_lock = threading.RLock()
_cache: dict[str, Any] | None = None
_cache_mtime: float = -1.0


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Nested config with dotted-path lookup: cfg.get('detector.confidence')."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def path(self, key: str) -> Path:
        """Resolve a config path value against the project root."""
        raw = str(self.get(key) or "")
        p = Path(raw)
        return p if p.is_absolute() else (ROOT / p)


def load(force: bool = False) -> Config:
    """Return the current config, re-reading config.yaml when it changes."""
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except OSError:
            mtime = -1.0

        if force or _cache is None or mtime != _cache_mtime:
            raw: dict[str, Any] = {}
            if CONFIG_PATH.exists():
                try:
                    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        raw = loaded
                except Exception:
                    # A malformed edit must not stop a 15-hour run.
                    raw = {}
            merged = _deep_merge(DEFAULTS, raw)

            # Cloud & Environment overrides
            db_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
            if db_url:
                merged.setdefault("supabase", {})["db_url"] = db_url

            sb_url = os.environ.get("SUPABASE_URL")
            if sb_url:
                merged.setdefault("supabase", {})["url"] = sb_url

            sb_key = (
                os.environ.get("SUPABASE_KEY")
                or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
                or os.environ.get("SUPABASE_ANON_KEY")
            )
            if sb_key:
                merged.setdefault("supabase", {})["key"] = sb_key

            sb_bucket = os.environ.get("SUPABASE_BUCKET")
            if sb_bucket:
                merged.setdefault("supabase", {})["bucket"] = sb_bucket

            nim_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
            if nim_key and not merged.get("labeler", {}).get("api_key"):
                merged.setdefault("labeler", {})["api_key"] = nim_key

            port = os.environ.get("PORT")
            if port and port.isdigit():
                merged.setdefault("server", {})["port"] = int(port)

            host = os.environ.get("HOST")
            if host:
                merged.setdefault("server", {})["host"] = host

            _cache = merged
            _cache_mtime = mtime

        return Config(_cache)



def _coerce(dotted: str, value: Any) -> Any:
    """Coerce an incoming value to the type of its default, or raise."""
    default = Config(DEFAULTS).get(dotted)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return str(value)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if text == "" or any(c in text for c in ":#{}[],&*?|>'\"%@`") or text != text.strip():
        return yaml.safe_dump(text, default_flow_style=True).strip().rstrip("...").strip()
    return text


def _rewrite_scalar(text: str, section: str, key: str, value: Any) -> tuple[str, bool]:
    """Replace one `section: -> key:` scalar in-place, keeping comments intact."""
    lines = text.splitlines(keepends=True)
    in_section = False
    section_re = re.compile(rf"^{re.escape(section)}\s*:\s*(#.*)?$")
    key_re = re.compile(rf"^(\s+{re.escape(key)}\s*:[ \t]*)([^#\n]*?)([ \t]*#.*)?$")

    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        if section_re.match(body):
            in_section = True
            continue
        if not in_section:
            continue
        # A non-indented, non-blank, non-comment line ends the section.
        if body.strip() and not body[0].isspace() and not body.lstrip().startswith("#"):
            in_section = False
            continue
        match = key_re.match(body)
        if match:
            prefix, comment = match.group(1), match.group(3) or ""
            lines[i] = f"{prefix}{_yaml_scalar(value)}{comment}\n"
            return "".join(lines), True
    return text, False


def update(changes: dict[str, Any]) -> Config:
    """Apply dotted-path changes to config.yaml and return the new config.

    Only keys in EDITABLE_KEYS are honoured; anything else is ignored, so a
    partially-valid payload still applies its valid half. Edits are written
    as targeted line replacements to preserve the file's documentation.
    """
    global _cache_mtime
    with _lock:
        original = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
        text = original
        merged = load(force=True).as_dict()
        applied: dict[str, Any] = {}
        needs_full_dump = False

        for dotted, raw_value in (changes or {}).items():
            if dotted not in EDITABLE_KEYS or "." not in dotted:
                continue
            try:
                value = _coerce(dotted, raw_value)
            except (TypeError, ValueError):
                continue

            section, key = dotted.split(".", 1)
            merged.setdefault(section, {})[key] = value
            applied[dotted] = value

            text, ok = _rewrite_scalar(text, section, key, value)
            if not ok:
                needs_full_dump = True

        if not applied:
            return load()

        if needs_full_dump:
            # Key absent from the file (hand-trimmed config): fall back to a
            # full serialisation. Comments are lost, values are correct.
            text = yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)

        tmp = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, CONFIG_PATH)
        _cache_mtime = -1.0  # force reload on next load()
        return load(force=True)
