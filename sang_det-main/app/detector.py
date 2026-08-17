"""License plate detection.

A dedicated plate detector - not OCR over full frames. On first run the model
weights are fetched from the Hugging Face Hub and cached in `models/`; after
that the tool is fully offline.

Device selection is automatic: CUDA -> Apple MPS -> CPU.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROOT, Config
from .logging_setup import get

log = get("detector")

MODELS_DIR = ROOT / "models"

# Tried in order. Each entry names a Hub repo; the first .pt inside it that
# looks like a plate detector is downloaded. Multiple candidates so one repo
# going away does not brick a fresh install.
HUB_CANDIDATES: list[tuple[str, str]] = [
    ("morsetechlab/yolov11-license-plate-detection", "model"),
    ("keremberke/yolov8n-license-plate", "model"),
    ("nakamura196/yolov8-license-plate", "model"),
]

# If the model has multiple classes, only these are treated as plates.
PLATE_CLASS_HINTS = ("plate", "license", "licence", "number_plate", "numberplate", "lp")


class DetectorError(RuntimeError):
    """Raised when no usable plate model can be loaded."""


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "plate"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@functools.lru_cache(maxsize=1)
def select_device(preference: str = "auto") -> str:
    """Resolve the compute device once per process."""
    preference = (preference or "auto").strip().lower()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise DetectorError("PyTorch is not installed") from exc

    if preference in ("cpu", "cuda", "mps") or preference.startswith("cuda:"):
        if preference.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to auto")
        elif preference == "mps" and not getattr(torch.backends, "mps", None):
            log.warning("MPS requested but unavailable; falling back to auto")
        else:
            return preference

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        log.info("Using CUDA device: %s", name)
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        log.info("Using Apple MPS device")
        return "mps"
    log.info("Using CPU (no GPU detected)")
    return "cpu"


def _local_model() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    weights = sorted(MODELS_DIR.glob("*.pt"))
    plate_first = [w for w in weights if any(h in w.name.lower() for h in PLATE_CLASS_HINTS)]
    for candidate in (plate_first + weights):
        if candidate.stat().st_size > 100_000:  # guard against truncated downloads
            return candidate
    return None


def _download_from_hub() -> Path | None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        log.warning("huggingface-hub not installed; cannot auto-download weights")
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, repo_type in HUB_CANDIDATES:
        try:
            files = [f for f in list_repo_files(repo_id, repo_type=repo_type) if f.endswith(".pt")]
            if not files:
                continue
            # Prefer an explicitly named plate/best weight, then the smallest
            # model variant (n < s < m < l), which is fast and accurate enough.
            files.sort(key=lambda f: (
                0 if any(h in f.lower() for h in PLATE_CLASS_HINTS) else 1,
                0 if "best" in f.lower() else 1,
                0 if "n." in f.lower() or "nano" in f.lower() else 1,
                len(f),
            ))
            log.info("Downloading plate detector: %s/%s", repo_id, files[0])
            path = hf_hub_download(
                repo_id=repo_id, filename=files[0], repo_type=repo_type,
                cache_dir=str(MODELS_DIR / ".hf_cache"),
            )
            target = MODELS_DIR / f"{repo_id.split('/')[-1]}.pt"
            try:
                target.write_bytes(Path(path).read_bytes())
                return target
            except OSError:
                return Path(path)
        except Exception as exc:
            log.warning("Could not fetch %s: %s", repo_id, exc)
    return None


def resolve_model_path(cfg: Config) -> Path:
    """Find plate weights: explicit config -> models/ -> Hugging Face Hub."""
    configured = str(cfg.get("detector.model_path") or "").strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
        raise DetectorError(f"detector.model_path does not exist: {path}")

    local = _local_model()
    if local:
        log.info("Using local weights: %s", local.name)
        return local

    downloaded = _download_from_hub()
    if downloaded:
        return downloaded

    raise DetectorError(
        "No license plate model found. Either connect to the internet for the "
        f"one-time auto-download, or drop an Ultralytics .pt plate detector "
        f"into {MODELS_DIR} and restart."
    )


class PlateDetector:
    """Thread-safe wrapper around an Ultralytics YOLO plate model."""

    def __init__(self, cfg: Config):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise DetectorError("ultralytics is not installed") from exc

        self.model_path = resolve_model_path(cfg)
        self.device = select_device(str(cfg.get("detector.device", "auto")))
        self._lock = threading.Lock()

        log.info("Loading %s on %s", self.model_path.name, self.device)
        self.model = YOLO(str(self.model_path))
        try:
            self.model.to(self.device)
        except Exception as exc:
            log.warning("Could not move model to %s (%s); staying on CPU", self.device, exc)
            self.device = "cpu"

        names = getattr(self.model, "names", {}) or {}
        self.class_names = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) \
            else {i: str(v) for i, v in enumerate(names)}
        self.plate_class_ids = self._plate_class_ids()
        log.info(
            "Model classes: %s (treating %s as plates)",
            list(self.class_names.values())[:8],
            "all" if self.plate_class_ids is None else
            [self.class_names[i] for i in self.plate_class_ids],
        )

        # fp16 is a straight win on CUDA; on MPS/CPU it is not reliably faster.
        self.half = bool(cfg.get("detector.half", True)) and self.device.startswith("cuda")
        self._warmup(cfg)

    def _plate_class_ids(self) -> set[int] | None:
        """None means single-class / all-classes-are-plates."""
        if len(self.class_names) <= 1:
            return None
        matches = {
            idx for idx, name in self.class_names.items()
            if any(hint in name.lower().replace(" ", "_") for hint in PLATE_CLASS_HINTS)
        }
        if matches:
            return matches
        log.warning(
            "Multi-class model with no plate-like class name; keeping all classes. "
            "Consider a dedicated plate model for cleaner output."
        )
        return None

    @staticmethod
    def _probe_image() -> np.ndarray:
        """A synthetic plate-like scene that reliably produces a detection.

        Warm-up must produce boxes, because the box path is what exercises
        torchvision's NMS kernel - the piece most likely to be broken.
        """
        image = np.full((720, 1280, 3), 45, dtype=np.uint8)
        image[250:620, 400:880] = (70, 72, 78)          # vehicle body
        image[517:575, 557:723] = (15, 15, 15)          # plate border
        image[520:572, 560:720] = (238, 238, 235)       # plate face
        for i in range(6):                              # dark character blocks
            x = 570 + i * 25
            image[534:560, x:x + 14] = (18, 18, 18)
        return image

    def _warmup(self, cfg: Config) -> None:
        """Pay the first-inference cost now, and prove the path actually works.

        This is not just a performance warm-up. A torch/torchvision build
        mismatch - e.g. torch `+cu124` alongside torchvision `+cpu` - leaves
        `torchvision::nms` unimplemented for CUDA. Inference then throws on
        every single frame, which the batch path would otherwise swallow into
        "zero detections": the run completes, every video is marked done, and
        the output folder is empty after 15 hours. Failing loudly here, and
        falling back to CPU, turns that into a warning and a slower run.
        """
        imgsz = int(cfg.get("detector.imgsz", 960))
        probe = self._probe_image()

        for candidate in (self.device, "cpu"):
            half = bool(cfg.get("detector.half", True)) and candidate.startswith("cuda")
            try:
                if candidate != self.device:
                    self.model.to(candidate)
                results = self.model.predict(
                    probe, imgsz=imgsz, conf=0.10, device=candidate,
                    half=half, verbose=False,
                )
                boxes = getattr(results[0], "boxes", None) if results else None
                found = 0 if boxes is None else len(boxes)
                self.device, self.half = candidate, half
                log.info(
                    "Detector ready on %s (warm-up found %d box(es))", candidate, found
                )
                if found == 0:
                    log.warning(
                        "Warm-up produced no detections. The model loaded and ran, "
                        "but may not be a license plate detector."
                    )
                return
            except Exception as exc:
                if candidate == "cpu":
                    raise DetectorError(
                        f"Inference fails on every device. Last error: {exc}"
                    ) from exc
                log.error(
                    "Inference is broken on %s (%s). Falling back to CPU - the run "
                    "will be slower but will actually produce output. If you have an "
                    "NVIDIA GPU, this usually means torch and torchvision are from "
                    "different builds; reinstall both with the same +cuXXX tag.",
                    candidate, exc,
                )
                self._free_vram()

    def detect_batch(self, images: list[np.ndarray], cfg: Config) -> list[list[Detection]]:
        """Run detection over a batch of BGR frames."""
        if not images:
            return []

        conf = float(cfg.get("detector.confidence", 0.35))
        imgsz = int(cfg.get("detector.imgsz", 960))
        iou = float(cfg.get("detector.iou", 0.45))
        max_det = int(cfg.get("detector.max_detections", 20))

        with self._lock:
            try:
                results = self.model.predict(
                    images, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                    device=self.device, half=self.half, verbose=False, stream=False,
                )
            except RuntimeError as exc:
                # Usually CUDA OOM. Drop to a per-frame pass so the run
                # continues at reduced throughput instead of dying.
                log.warning("Batched inference failed (%s); retrying frame-by-frame", exc)
                self._free_vram()
                results = []
                failures = 0
                first_error: Exception | None = None
                for image in images:
                    try:
                        results.extend(self.model.predict(
                            image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                            device=self.device, half=self.half, verbose=False,
                        ))
                    except Exception as inner:
                        failures += 1
                        first_error = first_error or inner
                        log.error("Frame inference failed: %s", inner)
                        results.append(None)

                # Every frame failing is a broken detector, not an empty road.
                # Raising marks the video errored instead of quietly "done"
                # with nothing saved.
                if failures == len(images):
                    raise DetectorError(
                        f"Inference failed on all {failures} frames of a batch: "
                        f"{first_error}"
                    ) from first_error

        out: list[list[Detection]] = []
        for result in results:
            out.append(self._parse(result) if result is not None else [])
        # Guard against a short results list so indices stay aligned to frames.
        while len(out) < len(images):
            out.append([])
        return out

    def _parse(self, result) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
        except Exception as exc:
            log.debug("Could not read boxes: %s", exc)
            return []

        detections: list[Detection] = []
        for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confs, classes):
            if self.plate_class_ids is not None and int(class_id) not in self.plate_class_ids:
                continue
            detections.append(
                Detection(
                    x1=int(round(float(x1))), y1=int(round(float(y1))),
                    x2=int(round(float(x2))), y2=int(round(float(y2))),
                    confidence=float(confidence),
                    label=self.class_names.get(int(class_id), "plate"),
                )
            )
        return detections

    def _free_vram(self) -> None:
        try:
            import torch

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def info(self) -> dict:
        return {
            "model": self.model_path.name,
            "device": self.device,
            "half": self.half,
            "classes": list(self.class_names.values()),
        }
