# 2stream_3D_CNN_Walk_Pytorch

Spinal disease gait classification project built with PyTorch Lightning, Hydra, and a two-view video pipeline.

## Overview

- Canonical video preparation code lives in `prepare_video/`
- Training and evaluation entrypoint is `project/main.py`
- Config is managed through `configs/config.yaml`
- Processed datasets are expected under `/workspace/data/balanced_data/ap` and `/workspace/data/balanced_data/lat`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

If you use the preprocessing pipeline, you also need:

- a YOLO model file such as `/workspace/project/yolo12n.pt`
- OpenCV-compatible video codecs

## Data Preparation

### 1. Raw data layout

`prepare_video preprocess` expects this structure:

```text
/workspace/data/raw/
  20240101/
    右足/
      above.mp4
      front.mp4
      side.mp4
    左足/
      above.mp4
      front.mp4
      side.mp4
    固定無し/
      above.mp4
      front.mp4
      side.mp4
```

The three videos in each session must have the same frame count and FPS.

### 2. Crop and segment raw videos

This step detects the subject, crops each frame, and creates synchronized segments.

```bash
python3 -m prepare_video preprocess \
  --input-root /workspace/data/raw \
  --output-root /workspace/data/segment \
  --model-path /workspace/project/yolo12n.pt
```

Output:

```text
/workspace/data/segment/
  20240101/
    right_leg/
      20240101_front_0000.mp4
      20240101_side_0000.mp4
      20240101_above_0000.mp4
    nothing/
      ...
```

### 3. Split into 5 folds

This step pairs `front` and `side` segments, then creates cross-validation datasets for `ap` and `lat`.

```bash
python3 -m prepare_video split \
  --input-root /workspace/data/segment \
  --output-root /workspace/data/5folds \
  --num-folds 5 \
  --clip-seconds 1.0
```

Output:

```text
/workspace/data/5folds/
  ap/fold0/train/normal/*.mp4
  ap/fold0/train/weight/*.mp4
  lat/fold0/train/normal/*.mp4
  lat/fold0/train/weight/*.mp4
```

### 4. Balance the folds

This step downsamples classes while preserving AP/LAT pairing.

```bash
python3 -m prepare_video balance \
  --input-root /workspace/data/5folds \
  --output-root /workspace/data/balanced_data
```

### 5. Optional rename utility

If you have older segment folders that still contain `ap_*.mp4` or `lat_*.mp4`, normalize them with:

```bash
python3 -m prepare_video rename /path/to/segment_root
```

## Training

Default training runs K-fold cross validation with the dataset defined in `configs/config.yaml`.

```bash
python3 project/main.py train.experiment=early_fusion
```

Other examples:

```bash
python3 project/main.py train.experiment=late_fusion
python3 project/main.py train.experiment=slow_fusion
python3 project/main.py train.experiment=single
```

Important config values:

- `data.processed_root`: processed dataset root
- `data.ap_data_path`: AP view dataset root
- `data.lat_data_path`: LAT view dataset root
- `train.fold`: number of folds
- `train.experiment`: `single`, `early_fusion`, `late_fusion`, `slow_fusion`

## Evaluation / Inference

`project/main.py` supports `fit` and `test`.

Run evaluation for a trained checkpoint:

```bash
python3 project/main.py \
  train.run_mode=test \
  train.experiment=early_fusion \
  train.current_fold=fold0 \
  train.ckpt_path=/absolute/path/to/checkpoint.ckpt
```

Notes:

- `train.current_fold` must match the fold used for the checkpoint
- test artifacts are saved under the current `train.log_path`
- paired-view experiments use both `ap` and `lat`; `single` uses `ap` only

## Outputs

Training logs:

```text
logs/<backbone>/<experiment>/<date>/<time>/
```

Evaluation outputs may include:

- TensorBoard logs
- saved predictions
- text metrics
- confusion matrix images

## Project Structure

```text
configs/
project/
prepare_video/
analysis/
```

`prepare_video/` is the only maintained video preprocessing package. The older duplicated folder layout has been removed.
