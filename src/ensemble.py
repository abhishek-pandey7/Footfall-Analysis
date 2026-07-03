"""CLIP-primary ensemble with v9.2 'uncertain' branch."""
from __future__ import annotations


def ensemble_gender(face_label: str, face_conf: float,
                    clip_gender: str, clip_ratio: float,
                    uncertain_ratio: float = 1.8,
                    uncertain_face_threshold: float = 0.70) -> str:
    """
    Decision tree:
      1. CLIP confident (ratio >= uncertain_ratio):
           - face agrees with CLIP  -> CLIP gender
           - face disagrees AND face confident -> face gender
           - else -> CLIP gender
      2. CLIP wishy-washy:
           - face confident -> face gender
           - else -> 'uncertain'   (v9.2 fix: was 'male' in v9)
    """
    clip_confident = clip_ratio >= uncertain_ratio
    face_confident = (face_conf >= uncertain_face_threshold
                      and face_label in ("male", "female"))
    if clip_confident:
        if face_confident and face_label != clip_gender:
            return face_label
        return clip_gender
    if face_confident:
        return face_label
    return "uncertain"
