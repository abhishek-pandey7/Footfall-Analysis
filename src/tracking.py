"""Tracker (IoU + size-scaled centroid) + ReIDBank."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .geometry import iou, centroid


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    last_seen: int
    missed: int = 0
    gender: str = "uncertain"
    clip_embedding: Optional[np.ndarray] = None
    face_label: str = "unknown"
    face_conf: float = 0.0
    clip_gender: str = "unknown"
    clip_ratio: float = 0.0
    first_seen: int = 0


class ReIDBank:
    """Stores CLIP embeddings of recently-expired tracks for short-term ReID."""

    def __init__(self, cosine_threshold: float = 0.75,
                 expiry_seconds: float = 30.0,
                 max_size: int = 200,
                 max_bank_size: Optional[int] = None):
        self.cos_threshold = float(cosine_threshold)
        self.expiry_seconds = float(expiry_seconds)
        self.max_size = int(max_bank_size if max_bank_size is not None else max_size)
        self._bank: List[Dict[str, Any]] = []

    def add(self, embedding, track_id, gender, fps, last_seen_frame):
        if embedding is None:
            return
        self._bank.append({
            "embedding": embedding.astype(np.float32),
            "track_id": track_id,
            "expired_wall_time": time.time(),
            "last_seen_frame": last_seen_frame,
            "gender": gender,
        })
        if len(self._bank) > self.max_size:
            self._bank.pop(0)

    def query(self, embedding) -> Optional[int]:
        if embedding is None or not self._bank:
            return None
        emb = embedding.astype(np.float32)
        emb = emb / max(np.linalg.norm(emb), 1e-6)
        now = time.time()
        best_id, best_sim = None, 0.0
        for rec in self._bank:
            if now - rec["expired_wall_time"] > self.expiry_seconds:
                continue
            e = rec["embedding"]
            e = e / max(np.linalg.norm(e), 1e-6)
            sim = float(np.dot(emb, e))
            if sim > best_sim:
                best_sim, best_id = sim, rec["track_id"]
        if best_id is not None and best_sim >= self.cos_threshold:
            self._bank = [r for r in self._bank if r["track_id"] != best_id]
            return best_id
        return None


class Tracker:
    def __init__(self, cfg: Dict[str, Any], reid: Optional[ReIDBank] = None):
        self.iou_thresh = float(cfg["iou_threshold"])
        self.centroid_max = float(cfg["centroid_max_dist"])
        self.persist = int(cfg["persist_frames"])
        self.size_scale = float(cfg["size_scale"])
        self.reid = reid
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections, frame_idx, clip_classifier=None):
        active = list(self.tracks.values())
        assigned: set = set()
        pairs = []
        for di, det in enumerate(detections):
            for tr in active:
                if tr.track_id in assigned:
                    continue
                i = iou(det["box"], tr.box)
                if i >= self.iou_thresh:
                    pairs.append((i, di, tr.track_id)); continue
                cd = _centroid_dist(det["box"], tr.box)
                size_factor = _size_ratio(det["box"], tr.box)
                if cd <= self.centroid_max and size_factor >= (1.0 - self.size_scale):
                    pairs.append((i - 0.5 * (cd / self.centroid_max), di, tr.track_id))
        pairs.sort(reverse=True)
        used_tracks = set()
        for _, di, tid in pairs:
            if di in assigned or tid in used_tracks:
                continue
            det = detections[di]; tr = self.tracks[tid]
            tr.box = det["box"]; tr.last_seen = frame_idx; tr.missed = 0
            if det.get("face_label") is not None:
                tr.face_label = det["face_label"]; tr.face_conf = det["face_conf"]
            if det.get("clip_gender") is not None:
                tr.clip_gender = det["clip_gender"]
                tr.clip_ratio = det["clip_ratio"]
                tr.gender = det["gender"]
                tr.clip_embedding = det.get("clip_embedding", tr.clip_embedding)
            assigned.add(di); used_tracks.add(tid)
        for di, det in enumerate(detections):
            if di in assigned:
                continue
            new_id = None
            if self.reid is not None and det.get("clip_embedding") is not None:
                new_id = self.reid.query(det["clip_embedding"])
            if new_id is None:
                new_id = self.next_id; self.next_id += 1
            self.tracks[new_id] = Track(
                track_id=new_id, box=det["box"], last_seen=frame_idx,
                first_seen=frame_idx, gender=det.get("gender", "uncertain"),
                face_label=det.get("face_label", "unknown"),
                face_conf=det.get("face_conf", 0.0),
                clip_gender=det.get("clip_gender", "unknown"),
                clip_ratio=det.get("clip_ratio", 0.0),
                clip_embedding=det.get("clip_embedding"),
            )
            assigned.add(di)
        for tid, tr in list(self.tracks.items()):
            if tr.last_seen < frame_idx:
                tr.missed += 1
                if tr.missed > self.persist:
                    if self.reid is not None and tr.clip_embedding is not None:
                        self.reid.add(tr.clip_embedding, tr.track_id,
                                      tr.gender, fps=1.0, last_seen_frame=tr.last_seen)
                    del self.tracks[tid]
        return self.tracks


def _centroid_dist(a, b):
    ax, ay = centroid(a); bx, by = centroid(b)
    return math.hypot(ax - bx, ay - by)


def _size_ratio(a, b):
    aa = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    ab = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    if ab <= 0:
        return 0.0
    return min(aa, ab) / max(aa, ab)
