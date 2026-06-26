import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from model import resnet20
from utils import set_seed, get_class_names, get_dataloaders
from config import RESULTS_DIR, PLOTS_DIR, BATCH_SIZE


def load_model(checkpoint_path, device):
    model = resnet20()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint.get('best_acc', None)


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix — CIFAR-10 Test Set')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_class_accuracy(class_acc, class_names, save_path):
    plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(class_names, class_acc, color=colors)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy on CIFAR-10 Test Set')
    plt.xticks(rotation=45)
    plt.ylim(0, 105)
    for bar, acc in zip(bars, class_acc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{acc:.1f}%',
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(all_probs, all_targets, class_names, save_path):
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        y_true = (all_targets == i).astype(int)
        y_score = all_probs[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Per Class (One-vs-Rest)')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    mean_auc = np.mean(list(roc_auc.values()))
    return mean_auc


def plot_precision_recall_curves(all_probs, all_targets, class_names, save_path):
    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    ap_scores = []

    for i in range(n_classes):
        y_true = (all_targets == i).astype(int)
        y_score = all_probs[:, i]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ap_scores.append(ap)
        plt.plot(recall, precision, lw=1.5, label=f'{class_names[i]} (AP={ap:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves — Per Class')
    plt.legend(loc='lower left', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    return np.mean(ap_scores)


def evaluate(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('  CIFAR-10 Evaluation — ResNet-20')
    print('=' * 70)

    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=4)
    print(f'  Test samples: {len(test_loader.dataset)}')

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f'  [ERROR] Checkpoint not found: {checkpoint_path}')
        print('  Run "python train.py" first.')
        return

    model, training_best = load_model(checkpoint_path, device)
    print(f'  Checkpoint: {checkpoint_path}')
    if training_best:
        print(f'  Training best val acc: {training_best:.2f}%')

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.concatenate(all_probs, axis=0)
    class_names = get_class_names()

    test_acc = accuracy_score(all_targets, all_preds)
    print(f'\n  Test Accuracy: {test_acc * 100:.2f}%')

    print('\n' + '=' * 70)
    print('  Classification Report')
    print('=' * 70)
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report)

    report_path = RESULTS_DIR / 'evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('CIFAR-10 Evaluation Report — ResNet-20\n')
        f.write(f'Test Accuracy: {test_acc * 100:.2f}%\n')
        f.write('=' * 70 + '\n')
        f.write(report)
        f.write('\n\n--- Additional Metrics ---\n')
        mean_auc = plot_roc_curves(all_probs, all_targets, class_names, PLOTS_DIR / 'roc_curves.png')
        mean_ap = plot_precision_recall_curves(all_probs, all_targets, class_names, PLOTS_DIR / 'precision_recall.png')
        f.write(f'Mean ROC-AUC (OvR): {mean_auc:.4f}\n')
        f.write(f'Mean Average Precision: {mean_ap:.4f}\n')

    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, PLOTS_DIR / 'confusion_matrix.png')

    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_targets == i
        if mask.any():
            per_class_acc[name] = 100. * (all_preds[mask] == i).sum() / mask.sum()

    plot_per_class_accuracy(
        [per_class_acc[name] for name in class_names],
        class_names,
        PLOTS_DIR / 'per_class_accuracy.png'
    )

    mean_auc = plot_roc_curves(all_probs, all_targets, class_names, PLOTS_DIR / 'roc_curves.png')
    mean_ap = plot_precision_recall_curves(all_probs, all_targets, class_names, PLOTS_DIR / 'precision_recall.png')

    print(f'\n  Mean ROC-AUC (One-vs-Rest): {mean_auc:.4f}')
    print(f'  Mean Average Precision: {mean_ap:.4f}')

    print('\n' + '-' * 70)
    print('  Per-Class Accuracy:')
    for name in class_names:
        print(f'    {name:<12s} {per_class_acc[name]:.2f}%')
    print('=' * 70)
    print(f'\nAll reports and plots saved to: {RESULTS_DIR}')


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Evaluation')
    parser.add_argument('--checkpoint', type=str, default='models/best.pt')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
