# Setup & Run Guide

This guide walks you through setting up the CIFAR-10 image classification project from scratch.

## Requirements

| Tool | Minimum Version | Check Command |
|------|----------------|---------------|
| Python | 3.9+ | `python --version` |
| pip | 21.0+ | `pip --version` |
| NVIDIA GPU | CUDA-capable | `nvidia-smi` |
| Git | 2.0+ | `git --version` |

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SiamFS/Cifar_10.git
cd Cifar_10
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python setup.py
```

This script will automatically:

| Action | Details |
|--------|---------|
| Detect GPU | Runs `nvidia-smi` to check if an NVIDIA GPU is available |
| Show GPU specs | Displays name, VRAM, driver version |
| Install PyTorch | GPU: installs CUDA 12.4 build. CPU: warns and asks confirmation |
| Install packages | numpy, matplotlib, seaborn, scikit-learn, gradio, tqdm, pandas |

**What if no GPU is found?** The script will warn you and ask if you want to continue with CPU-only PyTorch. Training on CPU is much slower (hours vs minutes), but still works.

### 5. Verify Installation

```bash
python model.py
```

Expected output:
```
resnet20: 0.27M params, output shape: torch.Size([2, 10])
```

## Training

### Quick Smoke Test (3 epochs, ~1 minute)

```bash
python train.py --epochs 3
```

Use this to verify everything works before committing to a full run.

### Full Training (200 epochs, ~30 minutes with GPU)

```bash
python train.py
```

**What happens during training:**
- The CIFAR-10 dataset is downloaded automatically (163 MB, first time only)
- Training and validation metrics are printed per epoch
- The best model checkpoint is saved to `models/best.pt`
- Full log is written to `results/training_log.txt`
- Training curves are saved to `results/plots/training_curves.png`

**Customize training:**

Edit `config.py` to change hyperparameters:
```python
NUM_EPOCHS = 200       # Number of training epochs
BATCH_SIZE = 128       # Images per batch
LEARNING_RATE = 0.1    # Initial learning rate
WEIGHT_DECAY = 5e-4    # L2 regularization
EARLY_STOP_PATIENCE = 25  # Stop if no improvement for N epochs
```

Or pass them as command-line arguments:
```bash
python train.py --epochs 100 --lr 0.05 --batch-size 256
```

## Evaluation

```bash
python evaluate.py
```

This runs the trained model on the held-out test set (10,000 images) and generates:

| Output | Location |
|--------|----------|
| Classification report | Console + `results/evaluation_report.txt` |
| Confusion matrix | `results/plots/confusion_matrix.png` |
| ROC curves (per class) | `results/plots/roc_curves.png` |
| Precision-Recall curves | `results/plots/precision_recall.png` |
| Per-class accuracy chart | `results/plots/per_class_accuracy.png` |

## Inference

### Web Application (Gradio)

```bash
python inference.py
```

Opens a web interface at `http://localhost:7860`. Upload any image to get predictions.

### Command Line

```bash
python inference.py path/to/image.jpg
```

Example output:
```
Image: path/to/image.jpg | Size: (800, 600)
---------------------------------------------
  1. automobile    98.45% <<< TOP
  2. truck          1.23%
  3. ship           0.12%
---------------------------------------------
```

## Generate Data Visualization Plots

```bash
python utils.py
```

Generates:
- `results/plots/data_split.png` — pie chart of train/val/test split
- `results/plots/class_distribution.png` — per-class distribution
- `results/plots/augmentations.png` — sample augmented training images

## Docker (Optional)

```bash
docker build -t cifar10-app .
docker run -p 7860:7860 cifar10-app
```

Then open `http://localhost:7860` in your browser.

## Results Location

```
results/
├── training_log.txt           # Full epoch-by-epoch console log
├── experiment_log.csv          # Structured CSV for analysis
├── evaluation_report.txt       # Final classification report
└── plots/
    ├── training_curves.png     # Loss & accuracy over epochs
    ├── confusion_matrix.png    # 10x10 confusion matrix
    ├── roc_curves.png          # ROC curves per class
    ├── precision_recall.png    # Precision-recall curves
    ├── per_class_accuracy.png  # Bar chart per class
    ├── data_split.png          # Train/val/test pie chart
    ├── class_distribution.png  # Class balance
    ├── lr_schedule.png         # Learning rate schedule
    └── augmentations.png       # Augmented sample images
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Install NVIDIA GPU drivers from https://nvidia.com/drivers |
| Out of memory | Reduce batch size: `python train.py --batch-size 64` |
| Model not trained | Run `python train.py` first |
| Checkpoint not found | Verify `models/best.pt` exists |
| Gradio won't start | Check port 7860 is not in use |
| Python version too old | Install Python 3.9+ from https://python.org |
