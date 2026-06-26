# Implementation Plan

## Dataset
- **Source:** `torchvision.datasets.CIFAR10` (auto-download, 163MB)
- **Train:** 50,000 → split 45,000 train / 5,000 validation (90/10)
- **Test:** 10,000 (1,000 per class, pre-separated by CIFAR-10)
- **Classes:** 10, perfectly balanced

## Files to Build (in order)

### 1. `requirements.txt`
```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
gradio
tqdm
pillow
pandas
```

### 2. `model.py` — ResNet Architecture
- `BasicBlock` — standard 3x3 conv + BN + ReLU residual block
- `CifarResNet` — CIFAR-adapted: 3x3 initial conv, 3 stages, no max pooling, no 7x7 conv
- Configurable depth: `resnet20()`, `resnet32()`, `resnet56()`
- Kaiming init + BatchNorm
- Input: `(B, 3, 32, 32)` → Output: `(B, 10)`

### 3. `utils.py` — Data Loading & Preprocessing
- `get_dataloaders(batch_size=128, num_workers=4)`:
  - Train: RandomCrop(32,padding=4) → RandomHorizontalFlip → RandAugment(N=2,M=14) → CutMix/MixUp → Normalize → ToTensor
  - Val/Test: Normalize → ToTensor only
- `get_class_names()` — ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
- `set_seed(42)` — reproducibility
- `visualize_augmentations()` — save sample augmented images grid to `results/plots/`
- `plot_data_distribution()` — pie/bar chart of class distribution

### 4. `train.py` — Training Loop
GPU setup:
```
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
scaler = GradScaler()          # Mixed precision
autocast(enabled=True)         # AMP
tensor.to(memory_format=torch.channels_last)
```

Train one epoch: forward → autocast → backward → scaler.step → scaler.update → log
Validate: forward → compute loss + accuracy
Early stopping: patience=30, save best model

**Verbose Logging (per epoch printed to console + saved to file):**
```
Epoch: 001/200 | LR: 0.1000 | Train Loss: 1.2345 | Train Acc: 56.78% | Val Loss: 1.0234 | Val Acc: 65.43% | Best: 65.43% | Time: 2.3s
```

**Saved outputs:**
- `models/best.pt` — best validation accuracy checkpoint
- `models/last.pt` — latest checkpoint (for resume)
- `results/training_log.txt` — full epoch-wise log (console output mirror)
- `results/experiment_log.csv` — structured log (epoch, lr, train_loss, val_loss, train_acc, val_acc)
- `results/plots/training_curves.png` — loss & accuracy curves
- `results/plots/lr_schedule.png` — learning rate over epochs

### 5. `evaluate.py` — Full Evaluation
- Load best model from checkpoint (`models/best.pt`)
- Run inference on test set (10,000 images)
- Metrics per class + macro/weighted avg:
  - Accuracy, Precision, Recall, F1-score
- Confusion matrix (heatmap) → `results/plots/confusion_matrix.png`
- Per-class accuracy bar chart → `results/plots/per_class_accuracy.png`
- Classification report printed to console + saved to `results/evaluation_report.txt`

### 6. `inference.py` — Gradio Web App
- Load model from `models/best.pt`
- CLI mode: `python inference.py dog.jpg` → prints top-3 predictions with confidence
- Web mode: `python inference.py` → Gradio at http://localhost:7860
- Auto-resize any input image to 32x32
- UI: image preview + top-3 predictions as horizontal bar chart + confidence percentages
- Error handling: invalid path, corrupt image, model not found → user-friendly error

### 7. `Dockerfile`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "inference.py"]
```

### 8. `.gitignore`
```
venv/
__pycache__/
*.pyc
models/*.pt
.env
.DS_Store
*.egg-info/
```

## Visualizations Checklist (for README)
| Plot | File | Add to README? |
|------|------|----------------|
| Training/Val Loss Curves | `results/plots/training_curves.png` | Yes |
| Training/Val Accuracy Curves | (combined in training_curves.png) | Yes |
| LR Schedule | `results/plots/lr_schedule.png` | Yes |
| Confusion Matrix | `results/plots/confusion_matrix.png` | Yes |
| Per-Class Accuracy | `results/plots/per_class_accuracy.png` | Yes |
| Sample Augmentations | `results/plots/augmentations.png` | Yes |
| Data Split Distribution | `results/plots/data_split.png` | Optional |

## Hyperparameter Grid Search
Run sequentially from train.py, log each experiment:
| Run | LR | WD | Optimizer | RandAug | CutMix |
|-----|----|-----|-----------|---------|--------|
| 1 | 0.1 | 5e-4 | SGD | N=2,M=14 | a=1.0 |
| 2 | 0.05 | 5e-4 | SGD | N=2,M=14 | a=1.0 |
| 3 | 0.1 | 1e-3 | SGD | N=2,M=14 | a=1.0 |
| 4 | 0.1 | 5e-4 | AdamW | N=2,M=14 | a=1.0 |

Results saved to `results/hyperparameter_tuning.csv`

## Edge Cases & Error Handling
| Scenario | Handling |
|----------|----------|
| No GPU available | Fallback to CPU with warning, skip AMP |
| Data download failure | Retry 3x, show error + manual download instructions |
| Decent CIFAR-10 download | torchvision handles automatically |
| OOM (Out of Memory) | Catch RuntimeError, reduce batch size by half, retry |
| NaN loss | Log warning, reduce LR by factor 10, skip epoch |
| Keyboard interrupt (Ctrl+C) | Save checkpoint before exit, allow resume |
| Model file not found | Show clear error: "Run train.py first" |
| Corrupt model checkpoint | Try safe loading (weights_only=True), fallback message |
| Invalid image for inference | Show "Unsupported file format" error |
| Empty image upload (Gradio) | Return "Please upload an image" |
| Results dirs missing | Auto-create `models/`, `results/`, `results/plots/` |
| Windows vs Linux paths | Use `pathlib.Path` throughout, no hardcoded separators |
| CuDNN non-deterministic | Set `torch.backends.cudnn.deterministic = True` for reproducibility |
| Multi-GPU detection | Use only first GPU (device 0) by default |

## Tests (Run Before Commit)
1. **Model:** instantiate resnet20, resnet56 → check param count matches expected, verify forward pass output shape
2. **Data:** load CIFAR-10, verify shapes (50000,3,32,32), check 10 classes, verify 90/10 split
3. **Augmentation:** visualize 16 augmented samples → `results/plots/augmentations.png`
4. **Training (smoke test):** 3 epochs on resnet20, verify loss decreases, verify AMP runs without NaN
5. **Checkpoint:** save/load model, verify weights match, test resume training
6. **Evaluation:** evaluate on test set, verify all metrics computed, confusion matrix non-empty
7. **Inference CLI:** `python inference.py examples/dog.jpg` — verify top-3 output
8. **Inference Gradio:** launch, verify UI opens at localhost:7860
9. **Edge cases:** corrupt image path, no GPU, interrupt training, missing checkpoint

## Execution Order
```
1. python setup.py              # Install dependencies + GPU detection
2. python utils.py --test       # Verify data loading + generate augmentation plot
3. python model.py --test       # Verify model instantiation + param counts
4. python train.py              # Full training + hyperparameter tuning
5. python evaluate.py           # Generate all evaluation plots + report
6. python inference.py          # Launch Gradio (web) or test CLI
7. ruff check .                 # Lint all code
```

## Results Directory Structure (after full run)
```
results/
├── training_log.txt               # Console log mirror (epoch by epoch)
├── experiment_log.csv             # Structured epoch metrics
├── hyperparameter_tuning.csv      # Grid search results
├── evaluation_report.txt          # Classification report + per-class metrics
├── plots/
│   ├── training_curves.png        # Loss + Accuracy curves (dual y-axis)
│   ├── lr_schedule.png            # LR vs epoch
│   ├── confusion_matrix.png       # 10x10 heatmap
│   ├── per_class_accuracy.png     # Bar chart per class
│   ├── augmentations.png          # 4x4 grid of augmented samples
│   └── data_split.png             # Train/Val/Test distribution
```

## Expected Timeline (RTX 4070)
| Step | Time |
|------|------|
| Setup + tests | 5 min |
| Train resnet20 (200 epochs) | ~8 min |
| Train resnet56 (200 epochs) | ~15 min |
| Hyperparameter grid (4 runs) | ~45 min total |
| Evaluate + plots | 1 min |
| **Total** | **~70 min** |
