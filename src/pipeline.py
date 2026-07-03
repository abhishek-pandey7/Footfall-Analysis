"""StoreAnalytics — orchestrates the whole pipeline (no age)."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from .config import load_config
from .geometry import dedup_boxes
from .models import YOLOPersonDetector, YuNetFaceDetector, FaceViTGender, CLIPGender
from .ensemble import ensemble_gender
from .tracking import Tracker, ReIDBank
from .counting import LineCounter, TimeBucketAggregator
from .reports import ReportGenerator
from .annotate import annotate_frame


class StoreAnalytics:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        print("[init] YOLO...", flush=True)
        self.detector = YOLOPersonDetector(cfg["detector"])
        print("[init] YuNet...", flush=True)
        self.face_det = YuNetFaceDetector(cfg["face_detector"])
        print("[init] FaceViT...", flush=True)
        self.face_gender = FaceViTGender(cfg["face_gender"])
        print("[init] CLIP...", flush=True)
        self.clip = CLIPGender(cfg["clip"])
        self.reid = None
        if cfg["reid"].get("enabled"):
            r = {k: v for k, v in cfg["reid"].items() if k != "enabled"}
            self.reid = ReIDBank(**r)
        self.tracker = Tracker(cfg["tracker"], reid=self.reid)
        self.line_counter = LineCounter(cfg["footfall"]["line"], cfg["footfall"]["direction"])
        self.agg = TimeBucketAggregator(cfg["aggregator"]["bucket_seconds"])
        self.uncertain_ratio = float(cfg["clip"]["uncertain_ratio"])
        self.uncertain_face_thresh = float(cfg["clip"]["uncertain_face_threshold"])
        self._gender_locked: set = set()
        self._seen_tracks: Dict[int, Dict[str, str]] = {}

    def process_video(self, video_path: str, out_dir: str,
                      dump_annotated_every: int = 30,
                      process_every_n: int = 1,
                      max_frames: Optional[int] = None,
                      start_frame: int = 1,
                      end_frame: Optional[int] = None) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = n_total
        if max_frames is not None:
            end_frame = min(end_frame, start_frame + max_frames - 1)
        self.line_counter.set_resolution(w, h)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dump_dir = out_path / "annotated_frames"
        dump_dir.mkdir(parents=True, exist_ok=True)

        print(f"[run] {video_path} {w}x{h} @ {fps:.2f}fps "
              f"frames {start_frame}..{end_frame} (of {n_total})", flush=True)

        if start_frame > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

        frame_idx = start_frame - 1
        processed = 0
        annotated_paths: List[str] = []
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx > end_frame:
                break
            if frame_idx % process_every_n != 0:
                continue

            boxes, scores = self.detector.detect(frame)
            if len(boxes) > 0:
                boxes, scores = dedup_boxes(boxes, scores, 0.45, 0.85)

            faces = self.face_det.detect(frame)

            dets: List[Dict[str, Any]] = []
            for bi, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                pad = 20
                cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad); cy2 = min(h, y2 + pad)
                body_crop = frame[cy1:cy2, cx1:cx2].copy()
                clip_out = self.clip.classify_body(body_crop)

                face_label = "unknown"; face_conf = 0.0
                best_face_area = 0; best_face_crop = None
                for (fbox, fscore, flm) in faces:
                    fx, fy = (fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        fa = (fbox[2] - fbox[0]) * (fbox[3] - fbox[1])
                        if fa > best_face_area:
                            best_face_area = fa
                            best_face_crop = frame[int(fbox[1]):int(fbox[3]),
                                                   int(fbox[0]):int(fbox[2])].copy()
                if best_face_crop is not None and best_face_crop.size > 0:
                    face_label, face_conf = self.face_gender.classify(best_face_crop)

                gender = ensemble_gender(
                    face_label, face_conf,
                    clip_out["gender"], clip_out["ratio"],
                    uncertain_ratio=self.uncertain_ratio,
                    uncertain_face_threshold=self.uncertain_face_thresh,
                )
                dets.append({
                    "box": box.astype(np.float32),
                    "score": float(scores[bi]),
                    "face_label": face_label, "face_conf": face_conf,
                    "clip_gender": clip_out["gender"], "clip_ratio": clip_out["ratio"],
                    "gender": gender,
                    "clip_embedding": clip_out["embedding"],
                })

            tracks = self.tracker.update(dets, frame_idx, clip_classifier=self.clip)
            for tid, tr in tracks.items():
                if tid in self._gender_locked:
                    tr.gender = self._seen_tracks[tid]["gender"]
                    continue
                if tr.gender in ("male", "female"):
                    self._gender_locked.add(tid)
                self._seen_tracks[tid] = {"gender": tr.gender}

            events = self.line_counter.update(tracks, frame_idx)
            t_sec = frame_idx / fps
            for ev in events:
                self.agg.add_event(t_sec, ev, tracks.get(ev["track_id"]))

            if dump_annotated_every > 0 and frame_idx % dump_annotated_every == 0:
                annotated = annotate_frame(frame, tracks)
                p = dump_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(p), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_paths.append(str(p))

            processed += 1
            if processed % 10 == 0:
                eta = (time.time() - t0) / processed * (end_frame - frame_idx) / process_every_n
                print(f"[run] frame {frame_idx}/{end_frame} "
                      f"tracks={len(tracks)} entries={self.line_counter.entries} "
                      f"exits={self.line_counter.exits} eta={eta:.0f}s", flush=True)

        cap.release()

        buckets = self.agg.finalise()
        gender_counts = {"male": 0, "female": 0, "uncertain": 0}
        for tid, info in self._seen_tracks.items():
            g = info["gender"] if info["gender"] in gender_counts else "uncertain"
            gender_counts[g] += 1
        unique_total = len(self._seen_tracks)

        report = ReportGenerator(out_dir)
        csv_path = report.write_daily_csv(buckets)
        per_person_path = report.write_per_person_csv(self._seen_tracks)
        pie_path = report.write_demographics_pie(
            gender_counts["male"], gender_counts["female"], gender_counts["uncertain"])
        foot_path = report.write_footfall_by_hour(buckets)
        summary = {
            "video": video_path,
            "fps": fps,
            "frames_processed": processed,
            "duration_seconds": processed / fps if fps > 0 else 0,
            "entries": self.line_counter.entries,
            "exits": self.line_counter.exits,
            "unique_track_estimate": unique_total,
            "gender_counts": gender_counts,
            "buckets": buckets,
            "reports": {
                "daily_report_csv": csv_path,
                "per_person_csv": per_person_path,
                "demographics_pie": pie_path,
                "footfall_by_hour": foot_path,
            },
            "annotated_frame_samples": annotated_paths[:10],
        }
        report.write_summary_json(summary)
        return summary
