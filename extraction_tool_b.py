"""
Tool B: CTR extraction via ianpan/chest-x-ray-basic
Adapted from Talha_wala_model.ipynb into an importable function.
"""

import cv2
import torch
import numpy as np
from transformers import AutoModel

from validation_gate import VisionResult

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

VIEW_LABELS = {0: "AP", 1: "PA", 2: "Lateral"}


def _load_model():
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
        _model = _model.eval().to(_device)
    return _model


def _calculate_ctr(mask: np.ndarray):
    """mask values: 1 = right lung, 2 = left lung, 3 = heart."""
    lungs = np.zeros_like(mask)
    lungs[mask == 1] = 1
    lungs[mask == 2] = 1
    heart = (mask == 3).astype("int")

    y, x = np.stack(np.where(lungs == 1))
    lung_min, lung_max = x.min(), x.max()

    y, x = np.stack(np.where(heart == 1))
    heart_min, heart_max = x.min(), x.max()

    lung_range = lung_max - lung_min
    heart_range = heart_max - heart_min
    return heart_range / lung_range


def _load_image(path: str, model):
    if path.lower().endswith((".dcm", ".dicom")):
        return model.load_image_from_dicom(path)
    img = cv2.imread(path, 0)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return img


def run_tool_b(image_path: str) -> dict:
    """
    Runs Tool B (ianpan/chest-x-ray-basic) on a single image.

    Returns a dict with: vision_result (VisionResult), view, age, sex,
    ctr (float or None), mask (np.ndarray), image (np.ndarray).
    """
    model = _load_model()
    img = _load_image(image_path, model)

    x = model.preprocess(img)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()

    with torch.inference_mode():
        out = model(x.to(_device))

    mask = out["mask"].argmax(1).squeeze(0).cpu().numpy()

    # The model's preprocess() resizes/pads the input to a fixed size, so the
    # predicted mask comes back in that resolution -- not the original image's.
    # Resize it back to the original image's shape (nearest-neighbor to keep
    # it a label map, not interpolated) so mask and image always align pixel
    # for pixel when overlaid.
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    view_idx = out["view"].argmax(1).item()
    view = VIEW_LABELS[view_idx]
    age = float(out["age"].item())
    is_female = bool(out["female"].item() >= 0.5)
    sex = "Female" if is_female else "Male"

    # Confidence: softmax max prob for view head, used as a simple proxy
    # for the safety gate's model-confidence check. Adjust if the model
    # exposes a dedicated segmentation-confidence score instead.
    view_probs = torch.softmax(out["view"], dim=1)
    confidence = float(view_probs.max().item())

    ctr = None
    if view != "Lateral":
        try:
            ctr = round(float(_calculate_ctr(mask)) * 100.0, 2)  # as a percentage
        except ValueError:
            ctr = None

    vision_result = VisionResult(ctr=ctr, confidence=confidence, tool_name="ToolB_ianpan")

    return {
        "vision_result": vision_result,
        "view": view,
        "age": round(age, 1),
        "sex": sex,
        "ctr": ctr,
        "mask": mask,
        "image": img,
    }
