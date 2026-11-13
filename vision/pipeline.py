"""
Two-stage dental vision pipeline (PyTorch .pt models).

Wraps YOUR trained models exactly as infer_gate.py / infer_detector.py do:
  Stage 1 gate     : EfficientNetV2-S (timm .pt) — "is this dental?"  P(dental) >= threshold
  Stage 2 detector : YOLO (Ultralytics .pt)      — 7 lesions + abnormal, conf 0.10

Heavy libs (torch/timm/ultralytics) are imported lazily inside the loaders, so
the rest of the app boots even before those are installed. If a model or dep is
missing, PipelineUnavailable is raised and the API returns a clear message.

Model locations (env overrides, else vision/ folder):
  GATE_WEIGHTS      default vision/gate_best.pt
  GATE_META         default vision/gate_meta.json
  DETECTOR_WEIGHTS  default vision/detector_best.pt
You can instead point these at your existing D:\\Dental paths.
"""
from __future__ import annotations

import os
import json

HERE = os.path.dirname(__file__)
GATE_WEIGHTS     = os.getenv("GATE_WEIGHTS",     os.path.join(HERE, "gate_best.pt"))
GATE_META        = os.getenv("GATE_META",        os.path.join(HERE, "gate_meta.json"))
DETECTOR_WEIGHTS = os.getenv("DETECTOR_WEIGHTS", os.path.join(HERE, "detector_best.pt"))
DETECTOR_CONF    = float(os.getenv("DETECTOR_CONF", "0.10"))
DETECTOR_IOU     = float(os.getenv("DETECTOR_IOU", "0.70"))
IMGSZ            = int(os.getenv("DETECTOR_IMGSZ", "640"))
CHUNK            = int(os.getenv("IMAGING_BATCH", "8"))   # images per model call (bounds memory)

# class 3 hypodontia is shown as "Edentulism" in deployment (model still outputs 3)
DISPLAY_RENAME = {"hypodontia": "Edentulism"}


class PipelineUnavailable(Exception):
    pass


_gate = None       # (model, meta, tf, device)
_detector = None   # YOLO


def _load_gate():
    global _gate
    if _gate is not None:
        return _gate
    if not os.path.exists(GATE_WEIGHTS) or not os.path.exists(GATE_META):
        raise PipelineUnavailable(
            f"gate model not found (looked for {GATE_WEIGHTS} and {GATE_META})")
    try:
        import torch, timm
        from torchvision import transforms
    except ImportError as e:
        raise PipelineUnavailable("install torch, torchvision, timm") from e

    meta = json.loads(open(GATE_META, encoding="utf-8").read())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(meta["model_name"], pretrained=False,
                              num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(GATE_WEIGHTS, map_location=device, weights_only=True))
    model.eval().to(device)
    size = int(meta.get("img_size", 224))
    tf = transforms.Compose([                       # exactly infer_gate.py's eval transform
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _gate = (model, meta, tf, device)
    return _gate


def gate_p_dental(pil_img):
    """Return (P(dental), threshold). Positive class index comes from meta."""
    model, meta, tf, device = _load_gate()   # friendly guard: models + deps
    import torch
    pos = meta["classes"].index(meta["positive_class"])   # 'dental' -> 0
    with torch.no_grad():
        x = tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
        p = float(torch.softmax(model(x), dim=1)[0][pos])
    return p, float(meta["threshold"])


def _load_detector():
    global _detector
    if _detector is not None:
        return _detector
    if not os.path.exists(DETECTOR_WEIGHTS):
        raise PipelineUnavailable(f"detector model not found ({DETECTOR_WEIGHTS})")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise PipelineUnavailable("install ultralytics") from e
    _detector = YOLO(DETECTOR_WEIGHTS)
    return _detector


def gate_p_dental_batch(pil_images):
    """Batched gate: one forward pass per CHUNK. Returns [(P(dental), threshold), ...]."""
    model, meta, tf, device = _load_gate()
    import torch
    pos = meta["classes"].index(meta["positive_class"])
    thr = float(meta["threshold"])
    out = []
    for i in range(0, len(pil_images), CHUNK):
        chunk = pil_images[i:i + CHUNK]
        xs = torch.stack([tf(im.convert("RGB")) for im in chunk]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(xs), dim=1)[:, pos].cpu().numpy().tolist()
        out.extend((float(p), thr) for p in probs)
    return out


def _decode(r, names):
    dets = []
    if r.boxes is not None and len(r.boxes):
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
            nm = names[int(c)]
            dets.append({"cls": int(c), "name": nm,
                         "display": DISPLAY_RENAME.get(nm, nm),
                         "conf": float(cf), "box": [float(v) for v in b]})
    return dets


def detect(pil_img):
    """Run the detector on one image. Returns list of detections."""
    model = _load_detector()
    r = model.predict(pil_img, imgsz=IMGSZ, conf=DETECTOR_CONF,
                      iou=DETECTOR_IOU, verbose=False)[0]
    return _decode(r, model.names)


def detect_batch(pil_images):
    """Batched detector: predicts CHUNK images per call. Returns [dets, dets, ...]."""
    if not pil_images:
        return []
    model = _load_detector()
    names = model.names
    out = []
    for i in range(0, len(pil_images), CHUNK):
        chunk = pil_images[i:i + CHUNK]
        for r in model.predict(chunk, imgsz=IMGSZ, conf=DETECTOR_CONF,
                               iou=DETECTOR_IOU, verbose=False):
            out.append(_decode(r, names))
    return out


# Video: sample frames rather than run every one. A few seconds at native
# fps is hundreds of near-duplicate frames — sampling at SAMPLE_FPS keeps
# inference cost bounded and, just as important, keeps the later per-class
# aggregation meaningful (see imaging_agent.analyze_video for why counting
# raw detections across every frame would wildly overcount).
SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "1.0"))
MAX_VIDEO_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "120"))


def extract_frames(video_path: str, sample_fps: float = None, max_frames: int = None):
    """Sample frames from a video at `sample_fps` frames/second (default
    SAMPLE_FPS), capped at `max_frames` (default MAX_VIDEO_FRAMES) so a long
    video can't blow up memory or inference time. Returns a list of PIL
    Images, in order."""
    try:
        import cv2
    except ImportError as e:
        raise PipelineUnavailable("install opencv-python-headless") from e
    from PIL import Image

    sample_fps = sample_fps or SAMPLE_FPS
    max_frames = max_frames or MAX_VIDEO_FRAMES

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise PipelineUnavailable(f"could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / sample_fps))

    frames, idx = [], 0
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        idx += 1
    cap.release()
    return frames
