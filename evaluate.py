import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model import resnet20
from utils import get_dataloaders, get_class_names, set_seed

RESULTS_DIR = Path('results')
PLOTS_DIR = RESULTS_DIR / 'plots'


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


def evaluate(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('  CIFAR-10 Evaluation — ResNet-20')
    print('=' * 70)

    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=4)
    print(f'  Test samples: {len(test_loader.dataset)}')

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f'  [ERROR] Checkpoint not found: {checkpoint_path}')
        print(f'  Run "python train.py" first.')
        return

    model, training_best = load_model(checkpoint_path, device)
    print(f'  Checkpoint: {checkpoint_path}')
    if training_best:
        print(f'  Training best val acc: {training_best:.2f}%')

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
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

    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, PLOTS_DIR / 'confusion_matrix.png')
    print(f'\nConfusion matrix saved to: {PLOTS_DIR / "confusion_matrix.png"}')

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
    print(f'Per-class accuracy plot saved to: {PLOTS_DIR / "per_class_accuracy.png"}')

    print('\n' + '-' * 70)
    print('  Per-Class Accuracy:')
    for name in class_names:
        print(f'    {name:<12s} {per_class_acc[name]:.2f}%')
    print('=' * 70)

    print(f'\nFull report saved to: {report_path}')
    return test_acc


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Evaluation')
    parser.add_argument('--checkpoint', type=str, default='models/best.pt')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
