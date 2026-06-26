import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, RANDOM_SEED, DATA_DIR,
    TRAIN_VAL_SPLIT, BATCH_SIZE, NUM_WORKERS, RANDAUG_NUM_OPS,
    RANDAUG_MAGNITUDE, RANDOM_ERASING_P, PLOTS_DIR,
)


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_names():
    return CLASS_NAMES


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=RANDAUG_NUM_OPS, magnitude=RANDAUG_MAGNITUDE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=RANDOM_ERASING_P),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=get_train_transform()
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=get_test_transform()
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=get_test_transform()
    )

    total_train = len(train_full)
    train_size = total_train - TRAIN_VAL_SPLIT
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_train))
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def plot_data_split(save_path):
    total = 60000
    train = 45000
    val = 5000
    test = 10000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sizes = [train, val, test]
    labels = ['Train (45,000)', 'Val (5,000)', 'Test (10,000)']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    explode = (0, 0.05, 0.05)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('CIFAR-10 Data Split Distribution')

    categories = ['Total Images', 'Train', 'Validation', 'Test']
    values = [total, train, val, test]
    bar_colors = ['#9b59b6', '#2ecc71', '#3498db', '#e74c3c']
    bars = ax2.bar(categories, values, color=bar_colors, edgecolor='white')
    ax2.set_ylabel('Number of Images')
    ax2.set_title('Dataset Split Sizes')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_class_distribution(save_path):
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    targets = np.array(dataset.targets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    class_counts = [(targets == i).sum() for i in range(10)]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    bars = ax1.barh(CLASS_NAMES, class_counts, color=colors, edgecolor='white')
    ax1.set_xlabel('Number of Images')
    ax1.set_title('Training Set Class Distribution (50,000 total)')
    for bar, count in zip(bars, class_counts):
        ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                 f'{count:,}', va='center', fontsize=9)

    per_class = [6000] * 10
    ax2.bar(CLASS_NAMES, per_class, color=colors, edgecolor='white')
    ax2.set_ylabel('Images per Class')
    ax2.set_title('CIFAR-10: Perfectly Balanced (6,000/class)')
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax2.axhline(y=6000, color='red', linestyle='--', alpha=0.5, label='6,000 baseline')
    ax2.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_augmentations(save_path, num_samples=16):
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=get_train_transform())
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        img = img * torch.tensor(CIFAR10_STD).view(3, 1, 1) + torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        ax.imshow(img)
        ax.set_title(CLASS_NAMES[label], fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


def print_model_architecture(model):
    from model import count_parameters
    params = count_parameters(model)
    print(f'\nModel: ResNet-20 | {params/1e6:.2f}M params')
    print('Input: (3, 32, 32)')
    print('Conv1: 3x3, 16 filters')
    print('Layer1: 3 blocks, 16 channels')
    print('Layer2: 3 blocks, 32 channels (stride=2)')
    print('Layer3: 3 blocks, 64 channels (stride=2)')
    print('Pool: AdaptiveAvgPool2d((1, 1))')
    print('FC: 64 -> 10')
    print('Output: (10,)')
    return params


if __name__ == '__main__':
    set_seed()
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f'Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}')

    plot_data_split(PLOTS_DIR / 'data_split.png')
    print(f'Data split plot saved: {PLOTS_DIR / "data_split.png"}')

    plot_class_distribution(PLOTS_DIR / 'class_distribution.png')
    print(f'Class distribution plot saved: {PLOTS_DIR / "class_distribution.png"}')

    visualize_augmentations(PLOTS_DIR / 'augmentations.png')
    print(f'Augmentation samples saved: {PLOTS_DIR / "augmentations.png"}')
