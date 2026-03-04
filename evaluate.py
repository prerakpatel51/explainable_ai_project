"""
evaluate.py - Evaluate a trained ResNet-50 on any DomainNet test set.

Supports in-domain (e.g., real->real) and cross-domain (e.g., real->sketch) evaluation.
All defaults loaded from config.yaml.

Usage:
    python evaluate.py                                     # uses config.yaml defaults
    python evaluate.py --test_domain sketch                # cross-domain eval
    python evaluate.py --checkpoint output/models/sketch/best_model.pt --test_domain sketch
"""

import argparse
import collections
import csv
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)
from torch.utils.data import DataLoader
from torchvision import models

from dataset import DomainNetDataset, get_transforms


def plot_training_curves(training_log_csv, eval_dir, train_domain):
    """Read training_log.csv and generate training curve plots."""
    if not os.path.isfile(training_log_csv):
        print(f"  Training log not found at {training_log_csv}, skipping training curve plots.")
        return

    # Read CSV
    epochs, train_loss, train_acc = [], [], []
    val_loss, val_acc = [], []
    val_f1_macro, val_f1_weighted = [], []
    val_precision, val_recall, lr_vals = [], [], []

    with open(training_log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
            val_f1_macro.append(float(row["val_f1_macro"]))
            val_f1_weighted.append(float(row["val_f1_weighted"]))
            val_precision.append(float(row["val_precision_macro"]))
            val_recall.append(float(row["val_recall_macro"]))
            lr_vals.append(float(row["lr"]))

    if not epochs:
        print("  Training log is empty, skipping training curve plots.")
        return

    domain_label = train_domain.capitalize()

    # --- Plot 1: Loss curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, "o-", label="Train Loss", markersize=3)
    ax.plot(epochs, val_loss, "s-", label="Val Loss", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{domain_label} Model - Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(eval_dir, "training_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Plot 2: Accuracy curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, "o-", label="Train Acc (%)", markersize=3)
    ax.plot(epochs, val_acc, "s-", label="Val Acc (%)", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{domain_label} Model - Training & Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(eval_dir, "training_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Plot 3: F1 scores ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_f1_macro, "o-", label="Val F1 (macro)", markersize=3)
    ax.plot(epochs, val_f1_weighted, "s-", label="Val F1 (weighted)", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{domain_label} Model - Validation F1 Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(eval_dir, "training_f1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Plot 4: Precision & Recall ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_precision, "o-", label="Val Precision (macro)", markersize=3)
    ax.plot(epochs, val_recall, "s-", label="Val Recall (macro)", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"{domain_label} Model - Validation Precision & Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(eval_dir, "training_precision_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Plot 5: Learning rate schedule ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, lr_vals, "o-", color="tab:green", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{domain_label} Model - Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(eval_dir, "training_lr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Plot 6: Combined summary (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{domain_label} Model - Training Summary", fontsize=14)

    axes[0, 0].plot(epochs, train_loss, "o-", label="Train", markersize=2)
    axes[0, 0].plot(epochs, val_loss, "s-", label="Val", markersize=2)
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_acc, "o-", label="Train", markersize=2)
    axes[0, 1].plot(epochs, val_acc, "s-", label="Val", markersize=2)
    axes[0, 1].set_title("Accuracy (%)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, val_f1_macro, "o-", label="Macro", markersize=2)
    axes[1, 0].plot(epochs, val_f1_weighted, "s-", label="Weighted", markersize=2)
    axes[1, 0].set_title("Val F1")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, val_precision, "o-", label="Precision", markersize=2)
    axes[1, 1].plot(epochs, val_recall, "s-", label="Recall", markersize=2)
    axes[1, 1].set_title("Val Precision & Recall")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=8)

    fig.tight_layout()
    path = os.path.join(eval_dir, "training_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


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
    eval_cfg = cfg["evaluation"]

    parser = argparse.ArgumentParser(description="Evaluate ResNet-50 on DomainNet")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--checkpoint", type=str, default=eval_cfg["checkpoint"],
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--test_domain", type=str, default=eval_cfg["test_domain"],
                        choices=["real", "sketch"],
                        help="Which domain's test set to evaluate on")
    parser.add_argument("--data_root", type=str, default=data_cfg["data_root"],
                        help="Root directory of XAI project")
    parser.add_argument("--batch_size", type=int, default=eval_cfg["batch_size"],
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=data_cfg["num_workers"],
                        help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default=eval_cfg["output_dir"],
                        help="Output directory")
    return parser.parse_args()


def load_split(filepath):
    """Load a split JSON file: list of [path, label] pairs."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [(item[0], item[1]) for item in data]


def balance_samples(samples, seed=42):
    """Subsample so every class has the same number of images (min across classes)."""
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))
    min_count = min(len(v) for v in by_class.values())
    balanced = []
    for label in sorted(by_class.keys()):
        items = by_class[label]
        rng.shuffle(items)
        balanced.extend(items[:min_count])
    return balanced, min_count


def main():
    args = parse_args()

    # Setup
    splits_dir = os.path.join(args.data_root, "splits")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  DomainNet ResNet-50 Evaluation")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Test domain:  {args.test_domain}")
    print(f"  Data root:    {args.data_root}")
    print(f"  Device:       {device}")
    print("=" * 60)

    # Load checkpoint
    print("\nLoading checkpoint ...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    train_args = checkpoint.get("args", {})
    train_domain = train_args.get("domain", "unknown")
    print(f"  Trained on domain: {train_domain}")
    print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"  Best val F1 (macro): {checkpoint.get('best_val_f1', 'N/A')}")
    print(f"  Number of classes: {num_classes}")

    # Build model
    print("\nBuilding ResNet-152 ...")
    model = models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("  Model loaded and set to eval mode")

    # Load test data
    test_split_path = os.path.join(splits_dir, f"{args.test_domain}_test.json")
    print(f"\nLoading test split: {test_split_path}")
    test_samples = load_split(test_split_path)
    print(f"  Test samples: {len(test_samples)}")

    test_dataset = DomainNetDataset(test_samples, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"  Test batches: {len(test_loader)}")

    # Run inference
    print("\nRunning inference ...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Batch [{batch_idx+1}/{len(test_loader)}]")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    overall_acc = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Overall Results ({train_domain} -> {args.test_domain})")
    print(f"{'='*60}")
    print(f"  Accuracy:          {overall_acc:.2f}%")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro):    {recall_macro:.4f}")

    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # Per-class accuracy
    acc_per_class = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            acc_per_class.append((all_preds[mask] == c).mean() * 100)
        else:
            acc_per_class.append(0.0)

    print(f"\n{'='*60}")
    print(f"  Per-Class Results")
    print(f"{'='*60}")
    print(f"  {'Class':<20} {'Acc%':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print(f"  {'-'*52}")
    for c in range(num_classes):
        print(f"  {class_names[c]:<20} {acc_per_class[c]:>7.2f}% "
              f"{f1_per_class[c]:>8.4f} {precision_per_class[c]:>8.4f} "
              f"{recall_per_class[c]:>8.4f}")

    # Save results
    eval_name = f"{train_domain}_model_on_{args.test_domain}"
    eval_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(eval_dir, exist_ok=True)

    # Save per-class metrics CSV
    per_class_csv = os.path.join(eval_dir, "per_class_metrics.csv")
    with open(per_class_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "accuracy", "f1", "precision", "recall"])
        for c in range(num_classes):
            writer.writerow([
                class_names[c],
                f"{acc_per_class[c]:.2f}",
                f"{f1_per_class[c]:.4f}",
                f"{precision_per_class[c]:.4f}",
                f"{recall_per_class[c]:.4f}",
            ])
    print(f"\nSaved per-class metrics: {per_class_csv}")

    # Save confusion matrix CSV (full test set)
    cm_full = confusion_matrix(all_labels, all_preds)
    cm_csv = os.path.join(eval_dir, "confusion_matrix.csv")
    with open(cm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for c in range(num_classes):
            writer.writerow([class_names[c]] + cm_full[c].tolist())
    print(f"Saved confusion matrix (full): {cm_csv}")

    # Build balanced test set (equal samples per class) for fair confusion matrix
    print("\nBuilding balanced test set for confusion matrix ...")
    balanced_samples, samples_per_class = balance_samples(test_samples)
    print(f"  Balanced: {len(balanced_samples)} samples "
          f"({samples_per_class} per class x {num_classes} classes)")

    bal_dataset = DomainNetDataset(balanced_samples, transform=get_transforms("test"))
    bal_loader = DataLoader(
        bal_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    bal_preds = []
    bal_labels = []
    with torch.no_grad():
        for images, labels in bal_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            bal_preds.extend(predicted.cpu().numpy())
            bal_labels.extend(labels.numpy())
    bal_preds = np.array(bal_preds)
    bal_labels = np.array(bal_labels)

    cm = confusion_matrix(bal_labels, bal_preds)

    # Per-class accuracy on balanced set (for label display)
    bal_acc_per_class = []
    for c in range(num_classes):
        mask = bal_labels == c
        if mask.sum() > 0:
            bal_acc_per_class.append((bal_preds[mask] == c).mean() * 100)
        else:
            bal_acc_per_class.append(0.0)

    # Select 20 representative classes for a readable confusion matrix:
    # top 15 by accuracy, 5 mid-range
    acc_arr = np.array(bal_acc_per_class)
    sorted_indices = np.argsort(acc_arr)  # ascending order

    top15 = sorted_indices[-15:][::-1].tolist()        # best 15
    # mid classes: pick 5 evenly spaced from the remaining pool
    mid_pool = sorted_indices[:-15]
    if len(mid_pool) >= 5:
        mid_step = len(mid_pool) / 5
        mid5 = [mid_pool[int(i * mid_step)] for i in range(5)]
    else:
        mid5 = mid_pool.tolist()

    selected = top15 + mid5
    seen = set()
    unique_selected = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            unique_selected.append(idx)
    selected_sorted = sorted(unique_selected)

    # Extract the sub-matrix
    cm_sub = cm[np.ix_(selected_sorted, selected_sorted)]
    sub_names = [class_names[i] for i in selected_sorted]
    sub_acc = [bal_acc_per_class[i] for i in selected_sorted]
    n_sub = len(selected_sorted)

    # Categorize for label coloring
    top_set = set(top15)

    # Save confusion matrix heatmap PNG (20-class subset, balanced)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_sub, interpolation="nearest", cmap="Blues")
    ax.set_title(
        f"Confusion Matrix: {train_domain.capitalize()} Model \u2192 "
        f"{args.test_domain.capitalize()} Test  "
        f"(20 classes, {samples_per_class} imgs/class)",
        fontsize=14, pad=20,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells with counts
    thresh = cm_sub.max() / 2
    for i in range(n_sub):
        for j in range(n_sub):
            val = cm_sub[i, j]
            if val != 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color="white" if val > thresh else "black")

    # Axis labels with accuracy annotation and color coding
    tick_marks = np.arange(n_sub)
    x_labels = [f"{name}" for name in sub_names]
    y_labels = [f"{name} ({acc:.0f}%)" for name, acc in zip(sub_names, sub_acc)]

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(y_labels, fontsize=8)

    # Color-code y-axis labels: green=top 15, orange=mid 5
    for i, orig_idx in enumerate(selected_sorted):
        color = "green" if orig_idx in top_set else "darkorange"
        ax.get_yticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_color(color)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True (per-class accuracy)", fontsize=11)

    plt.tight_layout()
    cm_png = os.path.join(eval_dir, "confusion_matrix.png")
    fig.savefig(cm_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix plot ({n_sub} classes): {cm_png}")

    # Save overall metrics
    overall_csv = os.path.join(eval_dir, "overall_metrics.csv")
    with open(overall_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", f"{overall_acc:.2f}"])
        writer.writerow(["f1_macro", f"{f1_macro:.4f}"])
        writer.writerow(["f1_weighted", f"{f1_weighted:.4f}"])
        writer.writerow(["precision_macro", f"{precision_macro:.4f}"])
        writer.writerow(["recall_macro", f"{recall_macro:.4f}"])
        writer.writerow(["train_domain", train_domain])
        writer.writerow(["test_domain", args.test_domain])
    print(f"Saved overall metrics: {overall_csv}")

    # Plot training curves from training_log.csv
    # Derive log path from checkpoint location: same directory as best_model.pt
    model_dir = os.path.dirname(args.checkpoint)
    training_log_csv = os.path.join(model_dir, "training_log.csv")
    print(f"\nGenerating training curve plots from: {training_log_csv}")
    plot_training_curves(training_log_csv, eval_dir, train_domain)

    print(f"\nAll evaluation results saved to: {eval_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
