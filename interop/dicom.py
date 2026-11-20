"""
DICOM import: intraoral cameras and X-ray sensors often save DICOM rather than
JPEG. Convert to an 8-bit image (applying the file's window/level and
photometric interpretation) and screen it with the imaging agent like any upload.

Uncompressed and most JPEG-compressed files work with pydicom alone; exotic
compressions (JPEG 2000, JPEG-LS) need pylibjpeg plugins — the error says so.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_image(path: str) -> tuple[Image.Image, dict]:
    import pydicom
    ds = pydicom.dcmread(path)
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise ValueError(f"cannot decode DICOM pixel data ({ds.file_meta.TransferSyntaxUID.name}): {e}") from e
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):   # multi-frame: take the first frame
        arr = arr[0]
    slope, intercept = float(getattr(ds, "RescaleSlope", 1) or 1), float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    center, width = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        c = float(center[0] if hasattr(center, "__len__") and not isinstance(center, str) else center)
        w = float(width[0] if hasattr(width, "__len__") and not isinstance(width, str) else width)
        lo, hi = c - w / 2, c + w / 2
    else:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":   # inverted greyscale
        arr = 255 - arr
    img = Image.fromarray(arr.astype(np.uint8))
    meta = {"modality": str(getattr(ds, "Modality", "")), "study_date": str(getattr(ds, "StudyDate", "")),
            "photometric": str(getattr(ds, "PhotometricInterpretation", "")),
            "size": list(img.size)}
    return img.convert("RGB"), meta


def screen(path: str, patient_id: int | None = None) -> dict:
    """Convert, then run the same two-stage vision pipeline as a photo upload."""
    import os
    import tempfile
    from agents import imaging_agent
    img, meta = to_image(path)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    try:
        img.save(tmp.name)
        result = imaging_agent.analyze_batch([tmp.name], patient_id)[0]
    finally:
        os.unlink(tmp.name)
    result["dicom"] = meta
    return result
