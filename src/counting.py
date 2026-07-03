"""Line crossing counter + time-bucket aggregator."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .geometry import centroid
from .tracking import Track


class LineCounter:
    def __init__(self, line_norm, direction="down"):
        self.line_norm = line_norm
        self.direction = direction
        self.prev_centroids: Dict[int, Tuple[float, float]] = {}
        self.last_state: Dict[int, int] = {}
        self.entries = 0
        self.exits = 0
        self.w = 1; self.h = 1

    def set_resolution(self, w, h):
        self.w = w; self.h = h

    def _line_pixels(self):
        (x1, y1), (x2, y2) = self.line_norm
        return ((x1 * self.w, y1 * self.h), (x2 * self.w, y2 * self.h))

    def _side(self, p):
        (lx1, ly1), (lx2, ly2) = self._line_pixels()
        cross = (lx2 - lx1) * (p[1] - ly1) - (ly2 - ly1) * (p[0] - lx1)
        return 1 if cross > 0 else (-1 if cross < 0 else 0)

    def update(self, tracks: Dict[int, Track], frame_idx: int):
        events = []
        for tid, tr in tracks.items():
            cur = centroid(tr.box)
            prev = self.prev_centroids.get(tid)
            self.prev_centroids[tid] = cur
            if prev is None:
                continue
            s_prev = self._side(prev); s_cur = self._side(cur)
            if s_prev == 0 or s_cur == 0 or s_prev == s_cur:
                continue
            direction = "down" if s_cur > s_prev else "up"
            last = self.last_state.get(tid)
            self.last_state[tid] = s_cur
            if last == s_cur:
                continue
            if direction == self.direction:
                self.entries += 1
                events.append({"frame": frame_idx, "track_id": tid, "type": "entry"})
            else:
                self.exits += 1
                events.append({"frame": frame_idx, "track_id": tid, "type": "exit"})
        return events


class TimeBucketAggregator:
    def __init__(self, bucket_seconds: int = 900):
        self.bucket_seconds = int(bucket_seconds)
        self.buckets: Dict[float, Dict[str, Any]] = defaultdict(
            lambda: {"entries": 0, "exits": 0,
                     "male": 0, "female": 0, "uncertain": 0,
                     "unique_tracks": set()})

    def add_event(self, t_sec, event, track: Optional[Track]):
        b = int(t_sec // self.bucket_seconds) * self.bucket_seconds
        rec = self.buckets[b]
        if event["type"] == "entry":
            rec["entries"] += 1
        else:
            rec["exits"] += 1
        if track is not None:
            rec["unique_tracks"].add(track.track_id)
            if track.gender in ("male", "female", "uncertain"):
                rec[track.gender] += 1

    def finalise(self):
        out = []
        for b in sorted(self.buckets.keys()):
            rec = self.buckets[b]
            out.append({
                "bucket_start": b,
                "entries": rec["entries"], "exits": rec["exits"],
                "male": rec["male"], "female": rec["female"], "uncertain": rec["uncertain"],
                "unique_tracks": len(rec["unique_tracks"]),
            })
        return out
