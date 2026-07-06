# Dataset download guide

The stage-1 PETA dataset requires manual download (academic access). After download, upload the zip to Colab via the `train_stage1.ipynb` upload cell — the notebook handles organization inline.

## PETA (PEdAttribute dataset)

- 19,000 pedestrian images, 8,700+ persons, 105 binary attributes
- **Where to request**: https://mmlab.ie.cuhk.edu.hk/projects/PETA.html
- **What you'll get**: a .zip with image folders + either `PETA.mat` or `Label.txt` files
- **Gender attribute**: `personalMale` (1=male, 0=female) in PETA.mat, or gender field in Label.txt

## Expected disk usage

| Dataset  | Images  | Disk   |
|----------|---------|--------|
| PETA     | ~19,000 | ~1.5 GB |
