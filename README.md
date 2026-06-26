# CIFAR-10 Image Classification

Deep learning image classification using PyTorch on the CIFAR-10 dataset. Built as part of the Trainee AI Engineer application for NITSOL Bangladesh Limited.

## Dataset

CIFAR-10: 60,000 32x32 color images across 10 classes.

| Split | Images | Description |
|-------|--------|-------------|
| Train | 45,000 | 90% of original training set (4,500/class) |
| Val | 5,000 | 10% of original training set (500/class) |
| Test | 10,000 | Pre-separated by CIFAR-10 (1,000/class) |

| Class | Label |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

## Quick Start

### 1. Setup
```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate  # Linux/Mac
python setup.py           # Auto-detects GPU, installs correct PyTorch
```

### 2. Train
```bash
python train.py
```

### 3. Evaluate
```bash
python evaluate.py
```

### 4. Inference (Web App)
```bash
python inference.py
# Opens Gradio web interface at http://localhost:7860
```

## Inference App

### Input
- Any image file (JPG, PNG, BMP) — auto-resized to 32x32
- Drag-and-drop or file upload via web UI

### Output
| Field | Description |
|-------|-------------|
| Predicted Class | Top-1 prediction (e.g. "dog") |
| Confidence | Probability score (e.g. 94.7%) |
| Top-3 | Top 3 predictions with confidence bars |

**Example:** Upload a photo of a frog → output shows "frog — 97.2%"

### Technology
- **Gradio** — web UI framework for ML demos
- **Docker** — containerized deployment
- No database or Redis needed (stateless inference)

## Docker
```bash
docker build -t cifar10-app .
docker run -p 7860:7860 cifar10-app
```

## Project Structure

```
Cifar_10/
├── README.md              # Project docs, setup, results (you are here)
├── AGENTS.md              # Developer/agent instructions
├── IMPLEMENTATION.md      # Full build plan, edge cases, tests
├── circular.txt           # Job circular
├── circular_extra.txt     # Extended circular details
├── setup.py               # Auto GPU detection + PyTorch install
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container for Hugging Face Spaces
├── .gitignore             # Ignore venv, __pycache__, checkpoints
│
├── model.py               # ResNet architecture (CIFAR-adapted)
├── train.py               # Training loop + hyperparameter grid search
├── evaluate.py            # Metrics, confusion matrix, plots
├── inference.py           # Gradio web UI + CLI fallback
├── utils.py               # Data loading, preprocessing, augmentation
│
├── models/                # Saved .pt checkpoints (gitignored)
├── results/
│   ├── training_log.txt   # Full epoch-by-epoch log
│   ├── experiment_log.csv # Structured metrics
│   ├── evaluation_report.txt
│   └── plots/             # All visualization PNGs
└── examples/              # Sample images for inference testing
```

## Model

Uses a CIFAR-adapted ResNet architecture with:
- 3x3 initial convolution (no 7x7 ImageNet conv)
- 3 residual stages
- Configurable depth (ResNet-20/32/44/56)

Training includes:
- RandAugment + CutMix/MixUp data augmentation
- Cosine annealing learning rate schedule
- Mixed precision (AMP) for GPU speedup
- Early stopping with checkpoint saving
- Full epoch-by-epoch logging to `results/training_log.txt`

## Training Output (per epoch)

```
Epoch: 001/200 | LR: 0.1000 | Train Loss: 1.2345 | Train Acc: 56.78% | Val Loss: 1.0234 | Val Acc: 65.43% | Best: 65.43% | Time: 2.3s
```

## Performance

### Model Comparison
| Experiment | Model | Params | Accuracy | Precision | Recall | F1 |
|------------|-------|--------|----------|-----------|--------|-----|
| 1 | ResNet-20 | 0.27M | TBD | TBD | TBD | TBD |
| 2 | ResNet-56 | 0.86M | TBD | TBD | TBD | TBD |

### Hyperparameter Tuning
| LR | WD | Optimizer | RandAug | CutMix | Val Acc |
|----|-----|-----------|---------|--------|---------|
| 0.1 | 5e-4 | SGD | N=2,M=14 | a=1.0 | TBD |
| 0.05 | 5e-4 | SGD | N=2,M=14 | a=1.0 | TBD |
| 0.1 | 1e-3 | SGD | N=2,M=14 | a=1.0 | TBD |
| 0.1 | 5e-4 | AdamW | N=2,M=14 | a=1.0 | TBD |

### Visualizations
| Plot | Description |
|------|-------------|
| ![Curves](results/plots/training_curves.png) | Training & validation loss/accuracy |
| ![LR](results/plots/lr_schedule.png) | Learning rate schedule |
| ![CM](results/plots/confusion_matrix.png) | Confusion matrix (10x10) |
| ![Class](results/plots/per_class_accuracy.png) | Per-class accuracy breakdown |
| ![Aug](results/plots/augmentations.png) | Sample augmented training images |

## License

MIT
