# WEC-UOD

Official implementation of **WEC-UOD: Structure-Enhanced Underwater Object Detection via Wavelet-Edge Collaboration and Selective Multi-Scale Fusion**.

WEC-UOD is an underwater object detector built on YOLOv11. It introduces a Wavelet-Edge Collaboration (WEC) module in the backbone and a Scale-Selective Fusion (SSF) module in the neck to improve structure-sensitive representation learning and multi-scale feature fusion under underwater degradation.

## Overview

Underwater images often suffer from scattering, absorption, blur, low contrast, and background interference. These degradations weaken object contours and fine texture details, making small-object detection more difficult.

WEC-UOD improves detector-internal feature learning without using an external image enhancement stage.

Main components:

- **WEC**: performs wavelet-subband compensation followed by edge-guided spatial refinement.
- **SSF**: selects informative multi-scale responses and refines fused features through channel-spatial recalibration.

## Repository Structure

```text
WEC-UOD/
├── configs/
│   ├── ruod.yaml
│   └── duo.yaml
├── models/
│   ├── wec_uod.yaml
│   └── modules/
│       ├── wec.py
│       └── ssf.py
├── requirements.txt
└── README.md
```

## Dataset Preparation

Please download the RUOD and DUO datasets from their official sources.

Organize the datasets in YOLO format:

```text
datasets/
├── RUOD/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── DUO/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Please update the dataset paths in:

```text
configs/ruod.yaml
configs/duo.yaml
```

## Training

Train WEC-UOD on RUOD:

```bash
python train.py --model models/wec_uod.yaml --data configs/ruod.yaml --img 640 --batch 32 --epochs 1000 --seed 0
```

Train WEC-UOD on DUO:

```bash
python train.py --model models/wec_uod.yaml --data configs/duo.yaml --img 640 --batch 32 --epochs 1000 --seed 0
```

## Evaluation

Evaluate on RUOD:

```bash
python val.py --weights runs/train/wec_uod_ruod/weights/best.pt --data configs/ruod.yaml --img 640
```

Evaluate on DUO:

```bash
python val.py --weights runs/train/wec_uod_duo/weights/best.pt --data configs/duo.yaml --img 640
```

## Main Settings

The main training settings used in the paper are:

```text
Input size: 640 × 640
Batch size: 32
Epochs: 1000
Initial learning rate: 0.01
Final learning rate factor: 0.01
Momentum: 0.937
Weight decay: 5e-4
Warm-up epochs: 3
Random seed: 0
```

All models were trained and evaluated using the same training schedule, augmentation configuration, and evaluation protocol.

## Citation

If this work is useful for your research, please cite our paper:

## Acknowledgement

This project is built upon the Ultralytics YOLO framework. We sincerely thank the Ultralytics team for their excellent open-source implementation.

