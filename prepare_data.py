"""
prepare_data.py - Select classes from DomainNet, create stratified splits,
                  balance per-class training counts across domains via oversampling,
                  and compute class weights.

Reads class list from config.yaml. Usage:
    python prepare_data.py                           # uses config.yaml defaults
    python prepare_data.py --config my_config.yaml   # custom config
    python prepare_data.py --data_root /other/path   # override data_root

After creating 80/10/10 splits for each domain, the training splits are balanced:
  - For each class, target_count = max(real_train_count, sketch_train_count)
  - The domain with fewer samples gets oversampled (duplicated entries) to match
  - Val/test splits are NOT modified (kept original for fair evaluation)
  - Existing training augmentation (RandomResizedCrop, flip, color jitter) ensures
    duplicated images look different each epoch

Outputs splits/ directory with JSON files for each domain's train/val/test sets,
class names, inverse-frequency class weights, and a balancing report.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_config(config_path="config.yaml"):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_samples(domain_dir, selected_classes, class_to_idx):
    """Collect all image paths and labels for selected classes in a domain directory."""
    samples = []
    skipped_classes = []

    for class_name in selected_classes:
        class_dir = os.path.join(domain_dir, class_name)
        if not os.path.isdir(class_dir):
            skipped_classes.append(class_name)
            continue

        label = class_to_idx[class_name]
        class_images = []
        for fname in os.listdir(class_dir):
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                class_images.append((os.path.join(class_dir, fname), label))

        samples.extend(class_images)

    if skipped_classes:
        print(f"  WARNING: Missing class directories: {skipped_classes}")

    return samples


def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency weights: weight_c = total / (num_classes * count_c)."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count_c = counts.get(c, 1)  # avoid division by zero
        weights.append(total / (num_classes * count_c))
    return weights


def create_splits(samples, test_size=0.1, val_size=0.1, random_state=42):
    """Create stratified train/val/test splits (80/10/10)."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # First split: 90% train+val, 10% test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    # Second split: from the 90%, take ~11.1% for val (= 10% of total)
    val_fraction = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_fraction,
        stratify=train_val_labels, random_state=random_state
    )

    train_set = list(zip(train_paths, train_labels))
    val_set = list(zip(val_paths, val_labels))
    test_set = list(zip(test_paths, test_labels))

    return train_set, val_set, test_set


def balance_training_splits(train_sets, selected_classes, seed=42):
    """Balance per-class counts across domain training splits via oversampling.

    For each class, finds the max count across all domains, then oversamples
    (duplicates entries randomly) for domains with fewer samples.

    Args:
        train_sets: dict of {domain: [(path, label), ...]}
        selected_classes: sorted list of class names
        seed: random seed for reproducible oversampling

    Returns:
        balanced_train_sets: dict of {domain: [(path, label), ...]}
        report: list of dicts with per-class balancing details
    """
    rng = random.Random(seed)
    num_classes = len(selected_classes)
    domains = list(train_sets.keys())

    # Group samples by class for each domain
    domain_class_samples = {}
    for domain in domains:
        by_class = {c: [] for c in range(num_classes)}
        for path, label in train_sets[domain]:
            by_class[label].append((path, label))
        domain_class_samples[domain] = by_class

    # Find target count per class (max across domains)
    target_counts = {}
    for c in range(num_classes):
        counts = {d: len(domain_class_samples[d][c]) for d in domains}
        target_counts[c] = max(counts.values())

    # Oversample and build report
    balanced = {d: [] for d in domains}
    report = []

    for c in range(num_classes):
        target = target_counts[c]
        row = {"class": selected_classes[c], "target": target}

        for domain in domains:
            original = domain_class_samples[domain][c]
            original_count = len(original)
            row[f"{domain}_original"] = original_count

            if original_count >= target:
                # No oversampling needed, take exactly target
                balanced[domain].extend(original[:target])
                row[f"{domain}_added"] = 0
            else:
                # Keep all originals + sample extra with replacement
                balanced[domain].extend(original)
                needed = target - original_count
                extras = [rng.choice(original) for _ in range(needed)]
                balanced[domain].extend(extras)
                row[f"{domain}_added"] = needed

        report.append(row)

    # Shuffle each domain's balanced set
    for domain in domains:
        rng2 = random.Random(seed + hash(domain))
        rng2.shuffle(balanced[domain])

    return balanced, report


def print_class_distribution(samples, domain_name, selected_classes):
    """Print per-class counts for a domain."""
    labels = [s[1] for s in samples]
    counts = Counter(labels)

    print(f"\n{'='*60}")
    print(f"  {domain_name} domain: {len(samples)} total images")
    print(f"{'='*60}")
    print(f"  {'Class':<20} {'Count':>8}")
    print(f"  {'-'*28}")

    for idx, class_name in enumerate(selected_classes):
        count = counts.get(idx, 0)
        print(f"  {class_name:<20} {count:>8}")

    print(f"  {'-'*28}")
    print(f"  {'TOTAL':<20} {len(samples):>8}")


def save_split(split_data, filepath):
    """Save a split as JSON: list of [path, label] pairs."""
    data = [[path, label] for path, label in split_data]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare DomainNet data splits")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data_root from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data"]["data_root"]

    # Read classes from config (sorted for deterministic label assignment)
    selected_classes = sorted(cfg["data"]["classes"])
    class_to_idx = {name: idx for idx, name in enumerate(selected_classes)}
    num_classes = len(selected_classes)

    splits_dir = os.path.join(data_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    print(f"Config file: {args.config}")
    print(f"Data root: {data_root}")
    print(f"Number of selected classes: {num_classes}")

    # ---- Phase 1: Create raw splits for each domain ----
    raw_train_sets = {}
    val_sets = {}
    test_sets = {}

    for domain in ["real", "sketch"]:
        domain_dir = os.path.join(data_root, domain)
        print(f"\n{'#'*60}")
        print(f"  Processing domain: {domain}")
        print(f"{'#'*60}")

        if not os.path.isdir(domain_dir):
            print(f"  ERROR: Domain directory not found: {domain_dir}")
            continue

        # Collect samples
        print(f"  Collecting images from {domain_dir} ...")
        samples = collect_samples(domain_dir, selected_classes, class_to_idx)
        print(f"  Found {len(samples)} images across {num_classes} classes")

        # Print distribution
        print_class_distribution(samples, domain, selected_classes)

        # Create splits
        print(f"\n  Creating stratified splits (80/10/10) ...")
        train_set, val_set, test_set = create_splits(samples)
        print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

        raw_train_sets[domain] = train_set
        val_sets[domain] = val_set
        test_sets[domain] = test_set

    # ---- Phase 2: Balance training splits across domains ----
    print(f"\n{'#'*60}")
    print(f"  Balancing training splits across domains")
    print(f"{'#'*60}")

    balanced_train_sets, balance_report = balance_training_splits(
        raw_train_sets, selected_classes
    )

    # Print balancing summary
    total_added = {"real": 0, "sketch": 0}
    print(f"\n  {'Class':<20} {'Target':>7} {'Real(orig)':>11} {'Real(+)':>8} "
          f"{'Sketch(orig)':>13} {'Sketch(+)':>9}")
    print(f"  {'-'*72}")
    for row in balance_report:
        r_added = row["real_added"]
        s_added = row["sketch_added"]
        total_added["real"] += r_added
        total_added["sketch"] += s_added
        print(f"  {row['class']:<20} {row['target']:>7} "
              f"{row['real_original']:>11} {r_added:>8} "
              f"{row['sketch_original']:>13} {s_added:>9}")

    print(f"\n  Oversampling summary:")
    for domain in ["real", "sketch"]:
        orig = len(raw_train_sets[domain])
        final = len(balanced_train_sets[domain])
        print(f"    {domain}: {orig} -> {final} (+{total_added[domain]} oversampled)")

    # ---- Phase 3: Save everything ----
    for domain in ["real", "sketch"]:
        # Save balanced training split
        save_split(balanced_train_sets[domain],
                   os.path.join(splits_dir, f"{domain}_train.json"))
        # Save val/test (unchanged)
        save_split(val_sets[domain],
                   os.path.join(splits_dir, f"{domain}_val.json"))
        save_split(test_sets[domain],
                   os.path.join(splits_dir, f"{domain}_test.json"))
        print(f"\n  Saved: {domain}_train.json, {domain}_val.json, {domain}_test.json")

        # Compute class weights from BALANCED training set
        train_labels = [s[1] for s in balanced_train_sets[domain]]
        class_weights = compute_class_weights(train_labels, num_classes)
        weights_path = os.path.join(splits_dir, f"{domain}_class_weights.json")
        with open(weights_path, "w") as f:
            json.dump(class_weights, f, indent=2)
        print(f"  Saved: {domain}_class_weights.json")

        # Print final balanced distribution
        print_class_distribution(balanced_train_sets[domain],
                                 f"{domain} (balanced train)", selected_classes)

    # Save class names (shared across domains)
    class_names_path = os.path.join(splits_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(selected_classes, f, indent=2)
    print(f"\nSaved: class_names.json")

    # Save balancing report
    report_path = os.path.join(splits_dir, "balancing_report.json")
    with open(report_path, "w") as f:
        json.dump(balance_report, f, indent=2)
    print(f"Saved: balancing_report.json")

    print(f"\nAll splits saved to: {splits_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
