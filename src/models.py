"""Model wrappers: YOLO person detector, YuNet face detector, FaceViT gender, CLIP gender (no age)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import urllib.request
import cv2
import numpy as np

from .geometry import validate_face_landmarks

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


class YOLOPersonDetector:
    def __init__(self, cfg: Dict[str, Any]):
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg["yolo_model"])
        self.conf = float(cfg["conf"])
        self.device = cfg.get("device", "cpu")
        self.imgsz = cfg.get("imgsz")
        self.min_w = float(cfg.get("min_box_w", 20))
        self.min_h = float(cfg.get("min_box_h", 30))
        self.ar_min = float(cfg.get("aspect_min", 0.15))
        self.ar_max = float(cfg.get("aspect_max", 6.0))

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kwargs = {"conf": self.conf, "classes": [0], "verbose": False, "device": self.device}
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        res = self.model.predict(frame_bgr, **kwargs)
        if not res or len(res[0].boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        b = res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        s = res[0].boxes.conf.cpu().numpy().astype(np.float32)
        keep = []
        for i, box in enumerate(b):
            w = float(box[2] - box[0]); h = float(box[3] - box[1])
            if w < self.min_w or h < self.min_h:
                continue
            ar = w / h
            if ar < self.ar_min or ar > self.ar_max:
                continue
            keep.append(i)
        if not keep:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return b[keep], s[keep]


class YuNetFaceDetector:
    """YuNet with v6 score bug FIXED (reads row[14], not row[4])."""
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        path = cfg["model_path"]
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            print(f"[YuNet] downloading to {path} ...", flush=True)
            urllib.request.urlretrieve(YUNET_URL, path)
        self.det = cv2.FaceDetectorYN.create(
            path, "", (320, 320),
            float(cfg["conf_threshold"]),
            float(cfg["nms_threshold"]),
            int(cfg["top_k"]),
        )

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        self.det.setInputSize((w, h))
        ok, faces = self.det.detect(frame_bgr)
        if not ok or faces is None:
            return []
        out = []
        for row in faces:
            x, y, ww, hh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score = float(row[14])  # v6 bug fix
            if score < self.cfg["conf_threshold"]:
                continue
            lm = np.array([
                [float(row[4]), float(row[5])],
                [float(row[6]), float(row[7])],
                [float(row[8]), float(row[9])],
                [float(row[10]), float(row[11])],
                [float(row[12]), float(row[13])],
            ], dtype=np.float32)
            xyxy = np.array([x, y, x + ww, y + hh], dtype=np.float32)
            if not validate_face_landmarks(lm, xyxy):
                continue
            out.append((xyxy, score, lm))
        return out


class FaceViTGender:
    """Face-based gender classifier (no age)."""
    def __init__(self, cfg: Dict[str, Any]):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch
        self.torch = torch
        self.cfg = cfg
        name = cfg["model"]
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.female_idx = self._find_idx(cfg["female_label"])
        self.male_idx = self._find_idx(cfg["male_label"])
        self.conf_thresh = float(cfg["conf_threshold"])

    def _find_idx(self, key):
        key = key.lower()
        for k, v in self.id2label.items():
            if key in str(v).lower():
                return int(k)
        raise RuntimeError(f"face ViT label '{key}' not found in {self.id2label}")

    def classify(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        from PIL import Image
        if face_bgr is None or face_bgr.size == 0:
            return "unknown", 0.0
        try:
            pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
            inputs = self.proc(images=pil, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[0]
            probs = self.torch.softmax(logits, dim=0)
            idx = int(probs.argmax().item())
            label = "female" if idx == self.female_idx else "male" if idx == self.male_idx else "unknown"
            return label, float(probs[idx].item())
        except Exception:
            return "unknown", 0.0


class CLIPGender:
    """CLIP zero-shot gender classifier (body-based). NO age."""
    def __init__(self, cfg: Dict[str, Any]):
        from transformers import CLIPModel, CLIPProcessor
        import torch
        self.torch = torch
        self.cfg = cfg
        name = cfg["model"]
        self.model = CLIPModel.from_pretrained(name)
        self.proc = CLIPProcessor.from_pretrained(name)
        self.model.eval()
        self.gender_text = cfg["gender_male_prompts"] + cfg["gender_female_prompts"]
        self.gender_text_emb = self._encode_text(self.gender_text)
        self.uncertain_ratio = float(cfg["uncertain_ratio"])

    def _encode_text(self, texts):
        with self.torch.no_grad():
            t = self.proc(text=texts, return_tensors="pt", padding=True)
            out = self.model.get_text_features(**t)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb.detach()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach()

    def _encode_image(self, pil_img):
        with self.torch.no_grad():
            inp = self.proc(images=pil_img, return_tensors="pt")
            out = self.model.get_image_features(**inp)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb.detach()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach()

    def classify_body(self, body_bgr: np.ndarray) -> Dict[str, Any]:
        """Returns {gender, gender_probs, ratio, embedding}."""
        from PIL import Image
        out = {"gender": "unknown",
               "gender_probs": {"male": 0.5, "female": 0.5},
               "ratio": 1.0, "embedding": None}
        if body_bgr is None or body_bgr.size == 0:
            return out
        try:
            pil = Image.fromarray(cv2.cvtColor(body_bgr, cv2.COLOR_BGR2RGB))
            img_emb = self._encode_image(pil)
            out["embedding"] = img_emb.cpu().numpy().flatten()
            g = (img_emb @ self.gender_text_emb.T).cpu().numpy()[0]
            n_male = len(self.cfg["gender_male_prompts"])
            male_score = float(np.mean(g[:n_male]))
            female_score = float(np.mean(g[n_male:]))
            mx = max(male_score, female_score)
            mn = max(min(male_score, female_score), 1e-6)
            ratio = mx / mn
            probs = np.exp([male_score, female_score])
            probs = (probs / probs.sum()).tolist()
            gender = "male" if male_score >= female_score else "female"
            out.update({
                "gender": gender,
                "gender_probs": {"male": float(probs[0]), "female": float(probs[1])},
                "ratio": float(ratio),
            })
        except Exception:
            pass
        return out
