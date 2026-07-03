"""Draw bounding boxes for verification (no age label)."""
from __future__ import annotations
from typing import Dict
import cv2
import numpy as np

from .tracking import Track

COLORS = {"male": (200, 80, 40), "female": (40, 80, 220), "uncertain": (120, 120, 120)}


def annotate_frame(frame_bgr: np.ndarray, tracks: Dict[int, Track]) -> np.ndarray:
    out = frame_bgr.copy()
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in tr.box]
        col = COLORS.get(tr.gender, (120, 120, 120))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"#{tid} {tr.gender}"
        if tr.face_conf > 0:
            label += f" face={tr.face_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
