# CIFAR-10 Image Classification

Deep learning image classification using PyTorch on CIFAR-10. Built for NITSOL Bangladesh Limited Trainee AI Engineer application.

## Results — ResNet-20 (0.27M params)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.88%** |
| Mean ROC-AUC (OvR) | 0.9969 |
| Mean Average Precision | 0.9788 |
| Precision (macro) | 92.87% |
| Recall (macro) | 92.88% |
| F1 (macro) | 92.87% |

### Per-Class Accuracy
| Class | Accuracy |
|-------|----------|
| airplane | 93.70% |
| automobile | 96.80% |
| bird | 90.60% |
| cat | 83.00% |
| deer | 94.80% |
| dog | 87.80% |
| frog | 95.60% |
| horse | 94.00% |
| ship | 96.10% |
| truck | 96.40% |

## Model Architecture
- **ResNet-20** adapted for CIFAR-10 (3x3 initial conv, no max pooling)
- 3 stages: 16 → 32 → 64 channels
- BatchNorm + Kaiming init
- 0.27M parameters

## Data Preprocessing
- **Train:** RandomCrop(32, padding=4) → RandomHorizontalFlip → RandAugment(N=2, M=14) → Normalize → RandomErasing(p=0.25)
- **Val/Test:** Normalize only
- Mean: (0.4914, 0.4822, 0.4465), Std: (0.2471, 0.2435, 0.2616)

## Training Config
| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch Size | 128 |
| Optimizer | SGD + momentum 0.9 + nesterov |
| Initial LR | 0.1 |
| Weight Decay | 5e-4 |
| Scheduler | CosineAnnealingLR |
| Early Stop | patience=25, min_delta=0.01 |
| AMP | Yes (Mixed Precision) |

All hyperparameters configurable via `config.py`.

## Quick Start
```bash
python -m venv venv && venv\Scripts\activate
python setup.py       # Auto GPU detection + install PyTorch
python train.py       # Train ResNet-20
python evaluate.py    # Full evaluation + all plots
python inference.py   # Gradio web UI
```

## Inference App
- **Gradio** web UI: upload any image → top-3 predictions with confidence
- Unknown class detection (confidence < 15%)
- Multiple object detection (top-2 gap < 15%)
- CLI mode: `python inference.py image.jpg`

## Visualizations
| Plot | Description |
|------|-------------|
| ![Curves](results/plots/training_curves.png) | Training & validation loss/accuracy |
| ![CM](results/plots/confusion_matrix.png) | Confusion matrix |
| ![ROC](results/plots/roc_curves.png) | ROC curves per class |
| ![PR](results/plots/precision_recall.png) | Precision-Recall curves |
| ![Class](results/plots/per_class_accuracy.png) | Per-class accuracy |
| ![Split](results/plots/data_split.png) | Data split distribution |
| ![Dist](results/plots/class_distribution.png) | Class distribution |
| ![LR](results/plots/lr_schedule.png) | LR schedule |
| ![Aug](results/plots/augmentations.png) | Augmentation samples |

## Docker
```bash
docker build -t cifar10-app .
docker run -p 7860:7860 cifar10-app
```

## Project Structure
```
Cifar_10/
├── config.py              # All hyperparameters (single source of truth)
├── model.py               # ResNet-20 architecture
├── utils.py               # Data loading, augmentation, plots
├── train.py               # Training loop with AMP + early stopping
├── evaluate.py            # Full evaluation + all metric plots
├── inference.py           # Gradio web UI + CLI mode
├── setup.py               # Auto GPU detection + install
├── requirements.txt
├── Dockerfile
├── .gitignore
├── models/                # Saved checkpoints
└── results/
    ├── training_log.txt   # Full epoch log
    ├── experiment_log.csv # CSV metrics
    ├── evaluation_report.txt
    └── plots/             # 9 visualization PNGs
```
