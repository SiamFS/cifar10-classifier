import os
import sys
import time
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import resnet20, resnet32, resnet56, count_parameters
from utils import set_seed, get_dataloaders, get_class_names

MODEL_DIR = Path('models')
RESULTS_DIR = Path('results')
PLOTS_DIR = RESULTS_DIR / 'plots'


def setup_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device('cuda:0')
        print(f'[GPU] {torch.cuda.get_device_name(0)} | TF32: ON | cuDNN benchmark: ON')
        return device, True
    else:
        print('[WARNING] No GPU found. Running on CPU.')
        return torch.device('cpu'), False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, filepath):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
    }
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def plot_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    lrs = [h['lr'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=1)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=1)
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(epochs, lrs, 'g-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'lr_schedule.png', dpi=150)
    plt.close()


def train(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = RESULTS_DIR / 'training_log.txt'

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log_file.write_text('', encoding='utf-8')
    log('=' * 70)
    log(f'  CIFAR-10 Training: {args.model}')
    log('=' * 70)
    log(f'  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    log(f'  Weight Decay: {args.weight_decay} | Optimizer: {args.optimizer}')
    log(f'  Scheduler: CosineAnnealingLR')
    log('-' * 70)

    set_seed(42)
    device, use_amp = setup_gpu()

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=4
    )
    log(f'  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}')
    log('-' * 70)

    model_fns = {'resnet20': resnet20, 'resnet32': resnet32, 'resnet56': resnet56}
    model = model_fns[args.model]()
    model = model.to(device)

    if use_amp and hasattr(args, 'channels_last') and args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        use_channels_last = True
    else:
        use_channels_last = False

    params = count_parameters(model)
    log(f'  Model: {args.model} | Params: {params/1e6:.2f}M | AMP: {use_amp} | Channels Last: {use_channels_last}')
    log('-' * 70)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9,
            weight_decay=args.weight_decay, nesterov=True
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    scaler = GradScaler() if use_amp else None

    best_acc = 0.0
    best_epoch = 0
    history = []
    start_time = time.time()

    csv_path = RESULTS_DIR / 'experiment_log.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = scheduler.get_last_lr()[0]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, MODEL_DIR / 'best.pt')

        history.append({
            'epoch': epoch, 'lr': current_lr,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
        })

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{current_lr:.6f}', f'{train_loss:.4f}', f'{train_acc:.2f}', f'{val_loss:.4f}', f'{val_acc:.2f}'])

        best_marker = ' *' if is_best else ''
        log(
            f'Epoch: {epoch:03d}/{args.epochs} | LR: {current_lr:.6f} | '
            f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
            f'Best: {best_acc:.2f}%{best_marker} | Time: {epoch_time:.1f}s'
        )

        if args.early_stop and epoch - best_epoch >= args.patience:
            log(f'\nEarly stopping triggered at epoch {epoch} (patience={args.patience})')
            break

    total_time = time.time() - start_time
    save_checkpoint(model, optimizer, scheduler, scaler, args.epochs, best_acc, MODEL_DIR / 'last.pt')

    log('-' * 70)
    log(f'  Training complete!')
    log(f'  Best Val Acc: {best_acc:.2f}% (epoch {best_epoch})')
    log(f'  Total Time: {total_time/60:.1f} min')
    log('=' * 70)

    plot_curves(history, PLOTS_DIR / 'training_curves.png')
    log(f'\nPlots saved to: {PLOTS_DIR}')

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet32', 'resnet56'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    parser.add_argument('--early-stop', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--no-early-stop', dest='early_stop', action='store_false')
    parser.add_argument('--no-channels-last', dest='channels_last', action='store_false')
    args = parser.parse_args()

    best = train(args)
    print(f'\nBest accuracy: {best:.2f}%')


if __name__ == '__main__':
    main()
