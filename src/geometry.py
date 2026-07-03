"""Box geometry + landmark validation."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    xa = max(float(a[0]), float(b[0])); ya = max(float(a[1]), float(b[1]))
    xb = min(float(a[2]), float(b[2])); yb = min(float(a[3]), float(b[3]))
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def contains(outer: np.ndarray, inner: np.ndarray, frac: float = 0.85) -> bool:
    xa = max(float(outer[0]), float(inner[0])); ya = max(float(outer[1]), float(inner[1]))
    xb = min(float(outer[2]), float(inner[2])); yb = min(float(outer[3]), float(inner[3]))
    if xb <= xa or yb <= ya:
        return False
    inter = (xb - xa) * (yb - ya)
    inner_area = max(1.0, (float(inner[2]) - float(inner[0])) * (float(inner[3]) - float(inner[1])))
    return (inter / inner_area) >= frac


def dedup_boxes(boxes, scores, iou_thresh=0.45, contain_thresh=0.85):
    """Greedy NMS that ALSO suppresses a smaller box when mostly inside a bigger one."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape[1] == 4 and boxes.max() <= 1.5:
        boxes = _xywh_to_xyxy(boxes)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)
    keep_idx: List[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep_idx.append(int(i))
        for j in order:
            if j == i or suppressed[j]:
                continue
            iou_ij = iou(boxes[i], boxes[j])
            area_i = _area(boxes[i]); area_j = _area(boxes[j])
            if area_i >= area_j:
                if contains(boxes[i], boxes[j], contain_thresh):
                    suppressed[j] = True; continue
            else:
                if contains(boxes[j], boxes[i], contain_thresh):
                    suppressed[i] = True
                    keep_idx.pop(); keep_idx.append(int(j)); break
            if iou_ij >= iou_thresh:
                if scores[j] < scores[i] or (scores[j] == scores[i] and j > i):
                    suppressed[j] = True
    keep_idx = list(dict.fromkeys(keep_idx))
    return boxes[keep_idx], scores[keep_idx]


def _xywh_to_xyxy(b):
    out = b.copy().astype(np.float32)
    out[:, 2] = b[:, 0] + b[:, 2]
    out[:, 3] = b[:, 1] + b[:, 3]
    return out


def _area(b):
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def centroid(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def validate_face_landmarks(landmarks, bbox):
    """landmarks: (5,2) [re, le, nose, rm, lm]. bbox: xyxy."""
    if landmarks is None or landmarks.shape != (5, 2):
        return False
    h = float(bbox[3]) - float(bbox[1])
    if h < 14.0:
        return False
    re, le, nose, rm, lm = landmarks
    if not (re[1] < nose[1] and le[1] < nose[1]):
        return False
    if not (nose[1] < rm[1] and nose[1] < lm[1]):
        return False
    eye_dx = abs(le[0] - re[0]); eye_dy = abs(le[1] - re[1])
    if eye_dx < 1e-3:
        return False
    if eye_dy / max(eye_dx, 1e-3) > 0.85:
        return False
    mouth_w = abs(lm[0] - rm[0])
    if mouth_w > 1.8 * eye_dx:
        return False
    if not (min(rm[0], lm[0]) - 2 <= nose[0] <= max(rm[0], lm[0]) + 2):
        return False
    return True
