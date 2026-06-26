import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
DATA_DIR = Path('data')


def set_seed(seed=42):
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
        transforms.RandAugment(num_ops=2, magnitude=14),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(batch_size=128, num_workers=4):
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=get_train_transform()
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=get_test_transform()
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=get_test_transform()
    )

    total_train = len(train_dataset)
    val_size = 5000
    train_size = total_train - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_train))

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def visualize_augmentations(save_path, num_samples=16):
    import matplotlib.pyplot as plt

    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=get_train_transform()
    )
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        img = img * torch.tensor(CIFAR10_STD).view(3, 1, 1) + torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        ax.imshow(img)
        ax.set_title(CLASS_NAMES[label], fontsize=8)
        ax.axis('off')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


if __name__ == '__main__':
    set_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}')
    print(f'Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}')

    images, labels = next(iter(train_loader))
    print(f'Batch shape: {images.shape}, Labels shape: {labels.shape}')
    print(f'Train augmentation: RandomCrop + RandomHorizontalFlip + RandAugment + RandomErasing + Normalize')
    print(f'Val/Test: Normalize only')
