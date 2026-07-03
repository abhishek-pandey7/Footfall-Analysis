"""CSV + PNG + JSON report writer (no age column)."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReportGenerator:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_daily_csv(self, buckets):
        path = self.out_dir / "daily_report.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bucket_start", "entries", "exits",
                        "male", "female", "uncertain", "unique_tracks"])
            for b in buckets:
                w.writerow([b["bucket_start"], b["entries"], b["exits"],
                            b["male"], b["female"], b["uncertain"], b["unique_tracks"]])
        return str(path)

    def write_per_person_csv(self, seen_tracks: Dict[int, Dict[str, str]]):
        path = self.out_dir / "per_person.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["track_id", "gender"])
            for tid, info in sorted(seen_tracks.items()):
                w.writerow([tid, info["gender"]])
        return str(path)

    def write_summary_json(self, summary):
        path = self.out_dir / "summary.json"
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        return str(path)

    def write_demographics_pie(self, male, female, uncertain):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        labels = ["Male", "Female", "Uncertain"]
        sizes = [male, female, uncertain]
        kept_l = [l for l, s in zip(labels, sizes) if s > 0]
        kept_s = [s for s in sizes if s > 0]
        if not kept_s:
            return None
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.pie(kept_s, labels=kept_l, autopct="%1.1f%%",
               colors=["#4C78A8", "#E45756", "#9AA0A6"], startangle=90)
        ax.set_title("Gender Demographics")
        path = self.out_dir / "demographics_pie.png"
        fig.savefig(path); plt.close(fig)
        return str(path)

    def write_footfall_by_hour(self, buckets):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        if not buckets:
            return None
        x = [b["bucket_start"] / 3600.0 for b in buckets]
        entries = [b["entries"] for b in buckets]
        exits = [b["exits"] for b in buckets]
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.bar([v - 0.1 for v in x], entries, width=0.2, label="entries", color="#4C78A8")
        ax.bar([v + 0.1 for v in x], exits, width=0.2, label="exits", color="#E45756")
        ax.set_xlabel("Hour"); ax.set_ylabel("Count")
        ax.set_title("Footfall by Hour"); ax.legend()
        path = self.out_dir / "footfall_by_hour.png"
        fig.savefig(path); plt.close(fig)
        return str(path)
