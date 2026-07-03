"""Default configuration + YAML loader (no age)."""
from __future__ import annotations
from typing import Any, Dict, Optional
import os
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "detector": {
        "yolo_model": "yolov8n.pt",
        "conf": 0.20,
        "device": "cpu",
        "imgsz": None,
        "min_box_w": 20,
        "min_box_h": 30,
        "aspect_min": 0.15,
        "aspect_max": 6.0,
    },
    "face_detector": {
        "model_path": "models/face_detection_yunet_2023mar.onnx",
        "conf_threshold": 0.45,
        "nms_threshold": 0.3,
        "top_k": 5000,
    },
    "face_gender": {
        "model": "rizvandwiki/gender-classification-2",
        "input_size": 224,
        "female_label": "female",
        "male_label": "male",
        "conf_threshold": 0.70,
    },
    "clip": {
        "model": "openai/clip-vit-base-patch32",
        "gender_male_prompts": ["a photo of a man", "a photo of a male person"],
        "gender_female_prompts": ["a photo of a woman", "a photo of a female person"],
        "uncertain_ratio": 1.8,
        "uncertain_face_threshold": 0.70,
    },
    "tracker": {
        "iou_threshold": 0.30,
        "centroid_max_dist": 80.0,
        "persist_frames": 3,
        "size_scale": 0.30,
    },
    "reid": {
        "enabled": True,
        "cosine_threshold": 0.75,
        "expiry_seconds": 30.0,
        "max_bank_size": 200,
    },
    "footfall": {
        "line": [[0.0, 0.7], [1.0, 0.7]],
        "direction": "down",
    },
    "aggregator": {
        "bucket_seconds": 900,
    },
    "pipeline": {
        "process_every_n_frames": 1,
        "max_frames": None,
        "dump_annotated_every": 30,
        "dump_dir": "annotated_frames",
    },
}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = _deep_copy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r") as fh:
            user = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user)
    return cfg


def _deep_copy(obj):
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return list(obj)
    return obj


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
