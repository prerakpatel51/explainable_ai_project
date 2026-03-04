"""
train.py - Train ResNet-50 on DomainNet (real or sketch domain).

Features:
- nn.DataParallel for multi-GPU
- Weighted CrossEntropyLoss (inverse-frequency)
- Checkpoint every epoch + best model saving
- CSV logging of per-epoch metrics
- Early stopping on validation loss
- Resume from checkpoint
- All defaults loaded from config.yaml

Usage:
    python train.py                                    # uses all config.yaml defaults
    python train.py --domain sketch                    # override domain only
    python train.py --resume output/models/real/checkpoints/epoch_15.pt
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models

from dataset import DomainNetDataset, get_transforms


def load_config(config_path="config.yaml"):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    # Pre-parse to get config path first
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    cfg = load_config(pre_args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    parser = argparse.ArgumentParser(description="Train ResNet-50 on DomainNet")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--domain", type=str, default=train_cfg["domain"],
                        choices=["real", "sketch"], help="Which domain to train on")
    parser.add_argument("--data_root", type=str, default=data_cfg["data_root"],
                        help="Root directory of XAI project")
    parser.add_argument("--epochs", type=int, default=train_cfg["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=train_cfg["batch_size"],
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=train_cfg["lr"],
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=train_cfg["patience"],
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=data_cfg["num_workers"],
                        help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=train_cfg["resume"],
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=train_cfg["output_dir"],
                        help="Output directory")
    return parser.parse_args()


def load_split(filepath):
    """Load a split JSON file: list of [path, label] pairs."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [(item[0], item[1]) for item in data]


def mixup_data(images, labels, alpha=0.2):
    """Apply mixup: blend pairs of images and return mixed images + label pairs + lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Compute mixup loss as weighted combination of two label losses."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """Train for one epoch with mixup augmentation, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Mixup
        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        # For accuracy tracking, count correct against the dominant label
        correct += (lam * predicted.eq(labels_a).sum().item()
                    + (1 - lam) * predicted.eq(labels_b).sum().item())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            batch_acc = 100.0 * correct / total
            print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} | Running Acc: {batch_acc:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model, return loss, accuracy, and per-class metrics."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics = {
        "val_loss": epoch_loss,
        "val_acc": epoch_acc,
        "val_f1_macro": f1_macro,
        "val_f1_weighted": f1_weighted,
        "val_precision_macro": precision_macro,
        "val_recall_macro": recall_macro,
    }
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_val_f1,
                    class_names, args, filepath):
    """Save training checkpoint."""
    # Unwrap DataParallel if needed
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "class_names": class_names,
        "args": vars(args),
    }
    torch.save(checkpoint, filepath)


def main():
    args = parse_args()

    # Setup
    domain = args.domain
    data_root = args.data_root
    splits_dir = os.path.join(data_root, "splits")
    output_dir = os.path.join(args.output_dir, "models", domain)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Print config
    print("=" * 60)
    print("  DomainNet ResNet-50 Training")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Domain:       {domain}")
    print(f"  Data root:    {data_root}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")
    print(f"  Patience:     {args.patience}")
    print(f"  Num workers:  {args.num_workers}")
    print(f"  Resume from:  {args.resume}")

    # Detect GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"  Device:       {device}")
    print(f"  GPUs:         {num_gpus}")
    if torch.cuda.is_available():
        for i in range(num_gpus):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)

    # Load class names and weights
    with open(os.path.join(splits_dir, "class_names.json"), "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"\nLoaded {num_classes} classes")

    with open(os.path.join(splits_dir, f"{domain}_class_weights.json"), "r") as f:
        class_weights = json.load(f)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Loaded class weights for {domain} domain")

    # Load splits
    print("\nLoading data splits ...")
    train_samples = load_split(os.path.join(splits_dir, f"{domain}_train.json"))
    val_samples = load_split(os.path.join(splits_dir, f"{domain}_val.json"))
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples:   {len(val_samples)}")

    # Create datasets and dataloaders
    train_dataset = DomainNetDataset(train_samples, transform=get_transforms("train"))
    val_dataset = DomainNetDataset(val_samples, transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Build model
    print("\nBuilding ResNet-152 (pretrained on ImageNet V2) ...")
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f"  Replaced fc layer: 2048 -> {num_classes}")

    # Freeze early layers (conv1, bn1, layer1, layer2) for first phase
    # Only layer3, layer4, and fc are trainable initially
    freeze_epochs = 5
    frozen_layers = ["conv1", "bn1", "layer1", "layer2"]
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in frozen_layers):
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen early layers for first {freeze_epochs} epochs")
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    # Wrap with DataParallel
    if num_gpus > 1:
        print(f"  Wrapping model with nn.DataParallel ({num_gpus} GPUs)")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    # Use different LR for pretrained vs new layers
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "fc" not in n]
    fc_params = [p for n, p in model.named_parameters()
                 if p.requires_grad and "fc" in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},   # lower LR for pretrained layers
        {"params": fc_params, "lr": args.lr},                # full LR for new fc head
    ], weight_decay=5e-4)

    # Cosine annealing with linear warmup (5 epochs)
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        # cosine decay from 1.0 to 0.01
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_f1 = 0.0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # Load model state (handle DataParallel wrapping)
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_val_f1 = checkpoint["best_val_f1"]
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, "
              f"best_val_f1={best_val_f1:.4f}")

    # CSV logging setup
    csv_path = os.path.join(output_dir, "training_log.csv")
    csv_fields = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "val_f1_macro", "val_f1_weighted", "val_precision_macro",
        "val_recall_macro", "lr", "time_sec"
    ]
    # Write header only if starting fresh (not resuming)
    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)
        print(f"\nCreated training log: {csv_path}")
    else:
        print(f"\nAppending to training log: {csv_path}")

    # Early stopping state
    epochs_without_improvement = 0
    # Count how many epochs without improvement happened before resume
    if args.resume and best_val_loss < float("inf"):
        # We don't know the exact count pre-resume, so reset to 0
        epochs_without_improvement = 0

    # Training loop
    print(f"\n{'='*60}")
    print(f"  Starting training from epoch {start_epoch + 1} to {args.epochs}")
    print(f"{'='*60}\n")

    layers_unfrozen = (start_epoch >= freeze_epochs)

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze all layers after freeze phase
        if not layers_unfrozen and epoch >= freeze_epochs:
            base_model = model.module if hasattr(model, "module") else model
            for param in base_model.parameters():
                param.requires_grad = True
            # Rebuild optimizer with all params (keeps fc LR higher)
            backbone_params = [p for n, p in model.named_parameters()
                               if "fc" not in n]
            fc_params = [p for n, p in model.named_parameters()
                         if "fc" in n]
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr * 0.1},
                {"params": fc_params, "lr": args.lr},
            ], weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            for _ in range(epoch):
                scheduler.step()
            layers_unfrozen = True
            print(f"\n  >>> Unfroze all layers at epoch {epoch+1}")

        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch+1}/{args.epochs} (lr={current_lr:.6f}) ---")

        # Train
        print("  [Train]")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        print("  [Validate]")
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_acc"]
        val_f1_macro = val_metrics["val_f1_macro"]
        val_f1_weighted = val_metrics["val_f1_weighted"]
        val_precision_macro = val_metrics["val_precision_macro"]
        val_recall_macro = val_metrics["val_recall_macro"]

        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Val F1 (macro): {val_f1_macro:.4f} | Val F1 (weighted): {val_f1_weighted:.4f}")
        print(f"  Val Precision (macro): {val_precision_macro:.4f} | Val Recall (macro): {val_recall_macro:.4f}")

        # Step scheduler (cosine annealing with warmup)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s")

        # Log to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.4f}", f"{train_acc:.2f}",
                f"{val_loss:.4f}", f"{val_acc:.2f}",
                f"{val_f1_macro:.4f}", f"{val_f1_weighted:.4f}",
                f"{val_precision_macro:.4f}", f"{val_recall_macro:.4f}",
                f"{current_lr:.6f}", f"{epoch_time:.1f}"
            ])

        # Save checkpoint every epoch
        ckpt_path = os.path.join(checkpoints_dir, f"epoch_{epoch+1:02d}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_val_f1,
                        class_names, args, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model (by macro F1)
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_path = os.path.join(output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_val_f1,
                            class_names, args, best_path)
            print(f"  New best model! Val F1 (macro): {best_val_f1:.4f} -> saved to {best_path}")

        # Early stopping (based on val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  No val loss improvement for {epochs_without_improvement}/{args.patience} epochs")

        if epochs_without_improvement >= args.patience:
            print(f"\n  Early stopping triggered after {args.patience} epochs without improvement.")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val F1 (macro): {best_val_f1:.4f}")
    print(f"  Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
