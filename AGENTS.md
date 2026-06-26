# AGENTS.md

## IMPORTANT: Always Read These Docs First
Before doing anything, read all `.md` and `.txt` files in the project root to understand full context and requirements.
Always read `IMPLEMENTATION.md` for the detailed build plan before writing any code.

## Project Overview
CIFAR-10 image classification project for NITSOL Bangladesh Limited Trainee AI Engineer application (Deadline: 26 June 2026).

**Reference Implementation:** chenyaofo/pytorch-cifar-models (BSD-3 license)

## Circular Requirements
1. Train & evaluate a Deep Learning model on CIFAR-10
2. Apply data preprocessing and augmentation techniques
3. Improve model performance through experimentation and optimization
4. Analyze and report model performance and accuracy
5. Develop a simple real-world inference application
6. Submit clean, documented, reproducible source code on GitHub with commit history

## Nice to Have (from circular)
- Kaggle-style experimentation logs
- Docker containerization
- Deployment on free hosting (Hugging Face Spaces)

## Environment
- OS: Windows
- Shell: PowerShell 5.1
- Python: 3.9+
- GPU: NVIDIA RTX 4070 (12GB VRAM)
- RAM: 32GB

## GPU Optimizations
- **Mixed Precision (AMP):** `torch.cuda.amp.autocast()` + `GradScaler` — 2-3x speedup, less VRAM
- **cuDNN:** `torch.backends.cudnn.benchmark = True`
- **DataLoader:** `pin_memory=True`, `persistent_workers=True`, `num_workers=4-8`
- **Memory format:** `tensor.to(memory_format=torch.channels_last)` — optimized for GPU tensor cores
- **TF32:** Enable on Ampere (RTX 4070) — `torch.backends.cuda.matmul.allow_tf32 = True`

## Commands

### Setup (Auto GPU Detection)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python setup.py
```
This auto-detects GPU via nvidia-smi, installs the correct PyTorch build (CUDA or CPU), and warns if no GPU found.

### Training
```powershell
python train.py
```

### Evaluation
```powershell
python evaluate.py
```

### Inference
```powershell
python inference.py <image_path>
```

### Linting
```powershell
pip install ruff; ruff check .
```

## Model Strategy
- Start with ResNet-20 (0.27M params) for fast iteration → scale to ResNet-56 (0.86M params, 94.37% baseline)
- ResNet architecture adapted for CIFAR: 3x3 initial conv, 3 stages, no max pooling

## Reference Hyperparameters (from chenyaofo/pytorch-cifar-models)
| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 128 |
| Optimizer | SGD + momentum 0.9 + nesterov |
| LR | 0.1 |
| Weight decay | 5e-4 |
| Scheduler | CosineAnnealingLR (T_max=200, eta_min=0) |
| Loss | CrossEntropyLoss |

## Dataset
- **Source:** `torchvision.datasets.CIFAR10` (auto-download, 163MB)
- **Train:** 50,000 → split 45,000 train / 5,000 validation (90/10)
- **Test:** 10,000 (1,000 per class, pre-separated by CIFAR-10)
- **Classes:** 10, perfectly balanced

## Data Preprocessing
- **Train:** RandomCrop(32, padding=4) + RandomHorizontalFlip + RandAugment + CutMix/MixUp + Normalize
- **Val/Test:** Normalize only
- Mean: (0.4914, 0.4822, 0.4465)
- Std: (0.2471, 0.2435, 0.2616)

## Training Output & Logging
- Console: per-epoch verbose output (epoch, LR, train/val loss, train/val acc, best acc, time)
- `results/training_log.txt` — full console log mirror
- `results/experiment_log.csv` — structured CSV for plotting
- `results/evaluation_report.txt` — classification report
- All plots saved to `results/plots/` (training curves, confusion matrix, per-class acc, LR schedule, augmentations)

## Hyperparameter Tuning Grid
| Parameter | Values |
|-----------|--------|
| Learning rate | 0.01, 0.05, 0.1 |
| Weight decay | 1e-4, 5e-4, 1e-3 |
| Optimizer | SGD + momentum vs AdamW |
| RandAugment N, M | (1, 5), (2, 9), (2, 14) |
| CutMix/MixUp alpha | 0.5, 1.0 |

## CIFAR-10 Benchmark Reference
| Model | Accuracy | Params |
|-------|----------|--------|
| resnet20 | 92.60% | 0.27M |
| resnet32 | 93.53% | 0.47M |
| resnet56 | 94.37% | 0.86M |
| repvgg_a1 | 94.89% | 12.82M |
| vgg13_bn | 94.22% | 28.33M |

## Inference App
- **Tech:** Gradio web UI (3 lines of code, no DB/Redis needed)
- **Input:** Any image file → auto-resized to 32x32
- **Output:** Top-3 predictions with confidence scores
- **CLI fallback:** `python inference.py <image_path>` for single image

## Deployment
- Docker + Gradio inference app
- Target: Hugging Face Spaces (free, no credit card, 24/7 uptime)
- No database, Redis, or external services required

## Conventions
- Use PyTorch for model implementation
- Follow PEP 8 style
- No unnecessary comments in code
- Keep code modular: separate files for model, training, evaluation, inference, utils
- Test before committing

## Security
- `.gitignore` must exclude: `venv/`, `__pycache__/`, `models/*.pt`, `*.pyc`, `.env`
- Never commit API keys, tokens, or passwords
- Gradio `server_name="0.0.0.0"` only in Docker (internal port); set `share=False` by default
- Model weights file (`models/*.pt`) tested only — no model poisoning risk
- `setup.py` uses hardcoded commands (nvidia-smi, pip install) — no injection vectors
