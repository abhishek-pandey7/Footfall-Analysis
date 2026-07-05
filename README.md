# Footfall Analysis — Retail CCTV Gender Analytics

Retail-store CCTV pipeline that detects people, classifies gender (male / female / uncertain), tracks them across frames, and counts footfall entries/exits. **Gender only — no age classification.**

## How to run

This repo is **notebook-driven**. There are no CLI scripts — everything runs from the two notebooks:

| Notebook | Purpose | Where |
|---|---|---|
| `gender_pipeline.ipynb` | **Inference** — full pipeline on a video (YOLO + YuNet face + gender ensemble + tracker + ReID + footfall counter + reports) | Colab or local Jupyter |
| `training_pipeline.ipynb` | **Training** — fine-tune ViT-B/16 on PETA+RAP+SCface (stage 1), then on your data (stage 2). Includes a built-in video test cell. | Colab (GPU recommended) |

## Quick start — inference (test a video)

1. Open `gender_pipeline.ipynb` in Colab or Jupyter.
2. Run all cells in order.
3. Upload your video when prompted (Section 10).
4. Outputs land in `outputs/run1/`: annotated frames, demographics pie chart, footfall chart, CSV reports, JSON summary.

## Quick start — training (fine-tune the model)

1. Open `training_pipeline.ipynb` in Colab (Runtime → T4 GPU).
2. Follow the cells in order:
   - Section 0-1: install deps, mount Drive
   - Section 3: upload + organize PETA / RAP / SCface datasets
   - Section 4: train stage 1 (~4-6 hours on T4)
   - Section 5: train stage 2 on your data (~30 min - 2 hours)
   - Section 6: quick eval on a test image
   - Cell 27 (after stage 1): **test on a video** — upload a video, get annotated output

3. Checkpoints auto-save to Google Drive at `/content/drive/MyDrive/IPD_checkpoints/`.

See `docs/training.md` for the full guide.

## Using a fine-tuned checkpoint in inference

After training, point the inference notebook at your checkpoint:

```python
# In gender_pipeline.ipynb, CONFIG cell:
CONFIG['face_gender']['model'] = '/content/checkpoints/stage1/best'
# or from Google Drive:
# CONFIG['face_gender']['model'] = '/content/drive/MyDrive/IPD_checkpoints/stage1'
```

## Do I need to re-run training if Colab runtime closes?

**No.** As long as the checkpoint was saved to Google Drive (the notebook does this automatically in the "copy to Drive" cell), you can:

1. Reopen `training_pipeline.ipynb` in Colab
2. Run cells 0-1 (install deps + mount Drive)
3. Skip directly to the section you need (e.g., stage 2, or the video test cell)
4. Load the checkpoint from Drive: `GenderClassifier.load('/content/drive/MyDrive/IPD_checkpoints/stage1')`

You do NOT need to re-upload datasets or re-run stage 1 training.

## Architecture

- **Backbone**: `google/vit-base-patch16-224` (pure ViT, ImageNet-21k pretrained, 768-dim features)
- **Task**: binary gender classification (no age)
- **Training**: two-phase (linear probe → full fine-tune) with cosine LR + warmup
- **Inference**: YOLOv8n person detection → YuNet face detection → ViT gender classification → ensemble → IoU+ReID tracker → line-crossing footfall counter

See `docs/architecture.md` for the full pipeline diagram and bug-fix history.

## Output files (in `outputs/<run>/`)

| File | Description |
|---|---|
| `daily_report.csv` | per-bucket entries/exits + gender counts |
| `per_person.csv` | one row per unique track ID with its gender |
| `demographics_pie.png` | overall gender mix (male/female/uncertain) |
| `footfall_by_hour.png` | entries vs exits per hour |
| `summary.json` | full machine-readable summary |
| `annotated_frames/frame_NNNNNN.jpg` | sampled annotated frames for VLM verification |

## Repo layout

```
gender_pipeline.ipynb        inference (full pipeline on video)
training_pipeline.ipynb      training (stage 1 + stage 2 + video test cell)
src/                         reference implementation (used by tests)
  config.py, geometry.py, models.py, ensemble.py, tracking.py,
  counting.py, reports.py, annotate.py, pipeline.py
  training/  model.py, datasets.py, trainer.py, transforms.py
tests/                       unit tests
docs/
  architecture.md            pipeline diagram + bug-fix table
  training.md                two-stage fine-tuning guide
scripts/
  download_datasets.md       manual download instructions for PETA/RAP/SCface
examples/
  config.example.yaml        reference for all tunables
outputs/                     generated reports + annotated frames (gitignored)
requirements.txt             pip dependencies
```

## Tests

```bash
pytest tests/ -v
```

## What's deliberately NOT in this version

- Age classification — will be added in a later version.
- CLI entry points — notebooks only (by design).
- CLIP fine-tuning — ViT only (CLIP is a VLM, less valuable for this task).
