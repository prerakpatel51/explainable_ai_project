"""
explain.py - XAI Explainability Pipeline for trained ResNet-50 models.

Applies 4 XAI methods (Grad-CAM, Grad-CAM++, Integrated Gradients, LIME)
to trained models and evaluates explanations along 4 axes:
  1. Stability
  2. Faithfulness (deletion/insertion)
  3. Cross-domain consistency
  4. Representation behavior (UMAP/t-SNE)

All defaults loaded from config.yaml.

Usage:
    python explain.py                                      # uses config.yaml defaults
    python explain.py --num_samples 2                      # override num_samples only
    python explain.py --checkpoint output/models/sketch/best_model.pt --output_dir output/xai/sketch_model
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from lime import lime_image
from PIL import Image
from scipy import ndimage
from skimage.segmentation import slic
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from umap import UMAP

from dataset import DomainNetDataset, get_transforms, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path="config.yaml"):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    # Pre-parse to get config path first
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    cfg = load_config(pre_args.config)
    data_cfg = cfg["data"]
    exp_cfg = cfg["explain"]

    parser = argparse.ArgumentParser(
        description="XAI Explainability Pipeline for ResNet-50"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--checkpoint", type=str, default=exp_cfg["checkpoint"],
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, default=data_cfg["data_root"],
                        help="Root directory of XAI project")
    parser.add_argument("--num_samples", type=int, default=exp_cfg["num_samples"],
                        help="Images per class per domain to explain")
    parser.add_argument("--output_dir", type=str, default=exp_cfg["output_dir"],
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=exp_cfg["batch_size"],
                        help="Batch size for IG and feature extraction")
    parser.add_argument("--ig_steps", type=int, default=exp_cfg["ig_steps"],
                        help="Interpolation steps for Integrated Gradients")
    parser.add_argument("--lime_samples", type=int, default=exp_cfg["lime_samples"],
                        help="LIME perturbation samples")
    parser.add_argument("--num_perturbations", type=int,
                        default=exp_cfg["num_perturbations"],
                        help="Perturbations per image for stability test")
    parser.add_argument("--num_workers", type=int, default=data_cfg["num_workers"],
                        help="DataLoader workers")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(filepath):
    """Load a split JSON file: list of [path, label] pairs."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [(item[0], item[1]) for item in data]


def sample_images(samples, num_per_class, seed=42):
    """Sample up to num_per_class images per class from a list of (path, label)."""
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))
    selected = []
    for label in sorted(by_class.keys()):
        items = by_class[label]
        rng.shuffle(items)
        selected.extend(items[:num_per_class])
    return selected


def load_image_tensor(path, device):
    """Load an image, apply test transforms, return tensor on device."""
    img = Image.open(path).convert("RGB")
    transform = get_transforms("test")
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def load_image_numpy(path):
    """Load image as numpy uint8 (224,224,3) after resize+center crop."""
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, 256)
    img = TF.center_crop(img, 224)
    return np.array(img, dtype=np.uint8)


def tensor_to_display(tensor):
    """Convert a normalized tensor (1,3,224,224) back to displayable numpy (224,224,3)."""
    img = tensor.squeeze(0).cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def normalize_heatmap(heatmap):
    """Normalize a heatmap to [0, 1]."""
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - hmin) / (hmax - hmin)


def save_attribution_png(original_np, heatmap, save_path):
    """Save 3-panel PNG: original | heatmap (jet) | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(original_np)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# XAI Method 1: Grad-CAM
# ---------------------------------------------------------------------------

def grad_cam(model, input_tensor, target_class, device):
    """Compute Grad-CAM attribution map for a single input."""
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = model.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.layer4.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    target_score = output[0, target_class]
    target_score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0].detach()  # (1, 2048, 7, 7)
    grad = gradients[0].detach()   # (1, 2048, 7, 7)

    # Global average pool gradients over spatial dims
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)
    cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    return normalize_heatmap(cam)


# ---------------------------------------------------------------------------
# XAI Method 2: Grad-CAM++
# ---------------------------------------------------------------------------

def grad_cam_pp(model, input_tensor, target_class, device):
    """Compute Grad-CAM++ attribution map for a single input."""
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = model.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.layer4.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    target_score = output[0, target_class]
    target_score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0].detach()  # (1, 2048, 7, 7)
    grad = gradients[0].detach()   # (1, 2048, 7, 7)

    # Grad-CAM++ weighting
    grad2 = grad.pow(2)
    grad3 = grad.pow(3)
    eps = 1e-8
    denom = 2.0 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True) + eps
    alpha = grad2 / denom
    weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)

    cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    return normalize_heatmap(cam)


# ---------------------------------------------------------------------------
# XAI Method 3: Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_gradients(model, input_tensor, target_class, device,
                         ig_steps=50, batch_size=32):
    """Compute Integrated Gradients attribution map for a single input."""
    # Baseline: zero tensor (ImageNet-mean in normalized space)
    baseline = torch.zeros_like(input_tensor)
    diff = input_tensor - baseline

    # Create interpolated inputs
    alphas = torch.linspace(0, 1, ig_steps + 1, device=device)
    all_grads = []

    # Process in batches
    for start in range(0, len(alphas), batch_size):
        end = min(start + batch_size, len(alphas))
        batch_alphas = alphas[start:end].view(-1, 1, 1, 1)
        interpolated = baseline + batch_alphas * diff  # (batch, 3, 224, 224)
        interpolated.requires_grad_(True)

        output = model(interpolated)
        scores = output[:, target_class].sum()
        scores.backward()
        all_grads.append(interpolated.grad.detach())

    # Average gradients (trapezoidal rule: endpoints weighted by 0.5)
    grads = torch.cat(all_grads, dim=0)  # (ig_steps+1, 3, 224, 224)
    avg_grads = (grads[:-1] + grads[1:]).sum(dim=0) / (2 * ig_steps)  # (3, 224, 224)

    # Attribution = avg_grads * (input - baseline), keep only positive contributions
    attribution = (avg_grads * diff.squeeze(0))          # (3, 224, 224), signed
    attribution = attribution.clamp(min=0).sum(dim=0)    # (224, 224)
    attribution = attribution.cpu().numpy()
    return normalize_heatmap(attribution)


# ---------------------------------------------------------------------------
# XAI Method 4: LIME
# ---------------------------------------------------------------------------

def lime_explanation(model, image_np, target_class, device, num_samples=1000):
    """Compute LIME attribution map for a single image."""
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def batch_predict(images):
        """Predict on a batch of numpy images (N, H, W, 3) uint8."""
        batch = torch.stack([normalize_transform(Image.fromarray(img))
                             for img in images])
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=42)
    segmentation_fn = lambda img: slic(img, n_segments=150, compactness=10, sigma=1, start_label=0)
    explanation = explainer.explain_instance(
        image_np,
        batch_predict,
        top_labels=None,
        labels=(target_class,),
        hide_color=128,  # gray fill avoids black-image distribution shift
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
        random_seed=42,
    )

    # Extract per-superpixel weights (signed: positive = supports class)
    segments = explanation.segments
    local_exp = explanation.local_exp[target_class]
    heatmap = np.zeros(segments.shape, dtype=np.float64)
    for seg_id, weight in local_exp:
        heatmap[segments == seg_id] = weight

    # Keep only positive contributions before normalizing so that
    # regions hurting the prediction don't get ranked above neutral ones.
    heatmap = np.clip(heatmap, 0, None)
    return normalize_heatmap(heatmap)


# ---------------------------------------------------------------------------
# Attribution map generation (Section C)
# ---------------------------------------------------------------------------

def generate_all_attributions(model, sampled_images, class_names, device, args,
                              output_dir):
    """Generate and cache attribution maps for all methods, domains, and images.

    Returns:
        cache: dict of {method: {domain: {idx: heatmap_np}}}
        image_info: dict of {domain: [(path, label, idx), ...]}
    """
    methods = ["gradcam", "gradcam_pp", "integrated_gradients", "lime"]
    domains = ["real", "sketch"]

    # Create output dirs
    for method in methods:
        for domain in domains:
            os.makedirs(os.path.join(output_dir, "attribution_maps", method, domain),
                        exist_ok=True)

    cache = {m: {d: {} for d in domains} for m in methods}
    image_info = {d: [] for d in domains}

    for domain in domains:
        items = sampled_images[domain]
        total = len(items)
        print(f"\n  Processing {domain} domain ({total} images) ...")

        for idx, (path, label) in enumerate(items):
            input_tensor = load_image_tensor(path, device)
            image_np = load_image_numpy(path)
            display_np = tensor_to_display(input_tensor)

            # Get model prediction for target class
            with torch.no_grad():
                logits = model(input_tensor)
                target_class = logits.argmax(dim=1).item()

            image_info[domain].append((path, label, target_class))
            fname = f"{class_names[label]}_{idx:04d}.png"

            # Grad-CAM
            hm = grad_cam(model, input_tensor, target_class, device)
            cache["gradcam"][domain][idx] = hm
            save_attribution_png(
                display_np, hm,
                os.path.join(output_dir, "attribution_maps", "gradcam", domain, fname),
            )

            # Grad-CAM++
            hm = grad_cam_pp(model, input_tensor, target_class, device)
            cache["gradcam_pp"][domain][idx] = hm
            save_attribution_png(
                display_np, hm,
                os.path.join(output_dir, "attribution_maps", "gradcam_pp", domain, fname),
            )

            # Integrated Gradients
            hm = integrated_gradients(model, input_tensor, target_class, device,
                                      ig_steps=args.ig_steps,
                                      batch_size=args.batch_size)
            cache["integrated_gradients"][domain][idx] = hm
            save_attribution_png(
                display_np, hm,
                os.path.join(output_dir, "attribution_maps",
                             "integrated_gradients", domain, fname),
            )

            # LIME
            hm = lime_explanation(model, image_np, target_class, device,
                                  num_samples=args.lime_samples)
            cache["lime"][domain][idx] = hm
            save_attribution_png(
                display_np, hm,
                os.path.join(output_dir, "attribution_maps", "lime", domain, fname),
            )

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"    [{domain}] {idx + 1}/{total} images done")

    return cache, image_info


# ---------------------------------------------------------------------------
# Evaluation Axis 1: Stability (Section D)
# ---------------------------------------------------------------------------

def perturb_tensor(input_tensor, device):
    """Apply a random perturbation: Gaussian noise or small rotation."""
    if random.random() < 0.5:
        # Gaussian noise (sigma=0.05 in normalized space)
        noise = torch.randn_like(input_tensor) * 0.05
        return (input_tensor + noise).to(device)
    else:
        # Small rotation ±5 degrees
        angle = random.uniform(-5, 5)
        rotated = TF.rotate(input_tensor.squeeze(0), angle).unsqueeze(0)
        return rotated.to(device)


def evaluate_stability(model, sampled_images, cache, image_info, device, args,
                       output_dir):
    """Evaluate stability of attribution methods under perturbations."""
    print("\n  Evaluating stability ...")
    methods = ["gradcam", "gradcam_pp", "integrated_gradients", "lime"]
    method_fns = {
        "gradcam": lambda inp, tc: grad_cam(model, inp, tc, device),
        "gradcam_pp": lambda inp, tc: grad_cam_pp(model, inp, tc, device),
        "integrated_gradients": lambda inp, tc: integrated_gradients(
            model, inp, tc, device, ig_steps=args.ig_steps,
            batch_size=args.batch_size),
    }
    domains = ["real", "sketch"]

    os.makedirs(os.path.join(output_dir, "stability"), exist_ok=True)
    rows = []

    for method in methods:
        for domain in domains:
            items = sampled_images[domain]
            ssim_scores = []

            for idx, (path, label) in enumerate(items):
                original_hm = cache[method][domain][idx]
                _, _, target_class = image_info[domain][idx]

                for _ in range(args.num_perturbations):
                    if method == "lime":
                        # Perturb in tensor space, then convert to numpy for LIME
                        input_tensor = load_image_tensor(path, device)
                        perturbed = perturb_tensor(input_tensor, device)
                        # Convert perturbed tensor back to numpy uint8
                        perturbed_display = tensor_to_display(perturbed)
                        perturbed_np = (perturbed_display * 255).clip(0, 255).astype(np.uint8)
                        perturbed_hm = lime_explanation(
                            model, perturbed_np, target_class, device,
                            num_samples=args.lime_samples,
                        )
                    else:
                        input_tensor = load_image_tensor(path, device)
                        perturbed = perturb_tensor(input_tensor, device)
                        perturbed_hm = method_fns[method](perturbed, target_class)

                    score = ssim(original_hm, perturbed_hm, data_range=1.0)
                    ssim_scores.append(score)

            mean_ssim = float(np.mean(ssim_scores))
            std_ssim = float(np.std(ssim_scores))
            rows.append([method, domain, f"{mean_ssim:.4f}", f"{std_ssim:.4f}"])
            print(f"    [{method}] {domain}: mean_ssim={mean_ssim:.4f} "
                  f"std_ssim={std_ssim:.4f}")

    csv_path = os.path.join(output_dir, "stability", "stability_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "domain", "mean_ssim", "std_ssim"])
        writer.writerows(rows)
    print(f"    Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Evaluation Axis 2: Faithfulness (Section E)
# ---------------------------------------------------------------------------

def evaluate_faithfulness(model, sampled_images, cache, image_info, device,
                          output_dir):
    """Evaluate faithfulness via deletion and insertion tests."""
    print("\n  Evaluating faithfulness ...")
    methods = ["gradcam", "gradcam_pp", "integrated_gradients", "lime"]
    domains = ["real", "sketch"]
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    os.makedirs(os.path.join(output_dir, "faithfulness"), exist_ok=True)
    deletion_rows = []
    insertion_rows = []

    for method in methods:
        for domain in domains:
            items = sampled_images[domain]
            del_aucs = []
            ins_aucs = []

            for idx, (path, label) in enumerate(items):
                heatmap = cache[method][domain][idx]
                _, _, target_class = image_info[domain][idx]
                input_tensor = load_image_tensor(path, device)

                # Sort pixel indices by attribution (high to low)
                flat_hm = heatmap.flatten()
                sorted_indices = np.argsort(flat_hm)[::-1]
                total_pixels = len(sorted_indices)

                # Baseline: zero tensor (ImageNet mean in normalized space)
                zero_tensor = torch.zeros_like(input_tensor)

                # --- Deletion test ---
                del_scores = []
                for p in percentages:
                    k = int(total_pixels * p)
                    mask = np.ones(total_pixels, dtype=np.float32)
                    mask[sorted_indices[:k]] = 0.0
                    mask_2d = mask.reshape(224, 224)
                    mask_tensor = torch.from_numpy(mask_2d).to(device).unsqueeze(0).unsqueeze(0)
                    masked = input_tensor * mask_tensor  # 0 = ImageNet mean in norm space
                    with torch.no_grad():
                        logits = model(masked)
                        prob = F.softmax(logits, dim=1)[0, target_class].item()
                    del_scores.append(prob)
                # AUC via trapezoidal rule (lower is more faithful)
                del_auc = np.trapz(del_scores, percentages)
                del_aucs.append(del_auc)

                # --- Insertion test ---
                ins_scores = []
                for p in percentages:
                    k = int(total_pixels * p)
                    mask = np.zeros(total_pixels, dtype=np.float32)
                    mask[sorted_indices[:k]] = 1.0
                    mask_2d = mask.reshape(224, 224)
                    mask_tensor = torch.from_numpy(mask_2d).to(device).unsqueeze(0).unsqueeze(0)
                    masked = zero_tensor + input_tensor * mask_tensor
                    with torch.no_grad():
                        logits = model(masked)
                        prob = F.softmax(logits, dim=1)[0, target_class].item()
                    ins_scores.append(prob)
                ins_auc = np.trapz(ins_scores, percentages)
                ins_aucs.append(ins_auc)

            mean_del = float(np.mean(del_aucs))
            std_del = float(np.std(del_aucs))
            mean_ins = float(np.mean(ins_aucs))
            std_ins = float(np.std(ins_aucs))

            deletion_rows.append([method, domain, f"{mean_del:.4f}", f"{std_del:.4f}"])
            insertion_rows.append([method, domain, f"{mean_ins:.4f}", f"{std_ins:.4f}"])
            print(f"    [{method}] {domain}: deletion_auc={mean_del:.4f} "
                  f"insertion_auc={mean_ins:.4f}")

    del_csv = os.path.join(output_dir, "faithfulness", "deletion_scores.csv")
    with open(del_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "domain", "mean_auc", "std_auc"])
        writer.writerows(deletion_rows)
    print(f"    Saved: {del_csv}")

    ins_csv = os.path.join(output_dir, "faithfulness", "insertion_scores.csv")
    with open(ins_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "domain", "mean_auc", "std_auc"])
        writer.writerows(insertion_rows)
    print(f"    Saved: {ins_csv}")


# ---------------------------------------------------------------------------
# Evaluation Axis 3: Cross-Domain Consistency (Section F)
# ---------------------------------------------------------------------------

def evaluate_cross_domain(cache, sampled_images, class_names, output_dir):
    """Evaluate cross-domain consistency of attribution maps."""
    print("\n  Evaluating cross-domain consistency ...")
    methods = ["gradcam", "gradcam_pp", "integrated_gradients", "lime"]

    os.makedirs(os.path.join(output_dir, "cross_domain"), exist_ok=True)
    rows = []

    # Build label-to-indices mapping for each domain
    domain_label_indices = {}
    for domain in ["real", "sketch"]:
        label_to_idx = collections.defaultdict(list)
        for idx, (path, label) in enumerate(sampled_images[domain]):
            label_to_idx[label].append(idx)
        domain_label_indices[domain] = label_to_idx

    for method in methods:
        for label in sorted(domain_label_indices["real"].keys()):
            real_indices = domain_label_indices["real"].get(label, [])
            sketch_indices = domain_label_indices["sketch"].get(label, [])

            if not real_indices or not sketch_indices:
                continue

            # Average heatmaps per domain to get class prototype
            real_maps = [cache[method]["real"][i] for i in real_indices]
            sketch_maps = [cache[method]["sketch"][i] for i in sketch_indices]

            real_proto = np.mean(real_maps, axis=0).flatten()
            sketch_proto = np.mean(sketch_maps, axis=0).flatten()

            # Cosine similarity
            dot = np.dot(real_proto, sketch_proto)
            norm_r = np.linalg.norm(real_proto)
            norm_s = np.linalg.norm(sketch_proto)
            if norm_r < 1e-8 or norm_s < 1e-8:
                cos_sim = 0.0
            else:
                cos_sim = float(dot / (norm_r * norm_s))

            rows.append([method, class_names[label], f"{cos_sim:.4f}"])

    csv_path = os.path.join(output_dir, "cross_domain",
                            "cross_domain_consistency.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "class_name", "cosine_similarity"])
        writer.writerows(rows)
    print(f"    Saved: {csv_path}")

    # Print per-method means
    method_sims = collections.defaultdict(list)
    for row in rows:
        method_sims[row[0]].append(float(row[2]))
    for method in methods:
        if method_sims[method]:
            mean_sim = np.mean(method_sims[method])
            print(f"    [{method}] mean cosine similarity: {mean_sim:.4f}")


# ---------------------------------------------------------------------------
# Evaluation Axis 4: Representation Behavior (Section G)
# ---------------------------------------------------------------------------

def extract_features(model, sampled_images, device, batch_size=32):
    """Extract 2048-dim features from model.avgpool for all sampled images."""
    print("\n  Extracting features ...")
    features_list = []
    labels_list = []
    domains_list = []

    feature_store = []

    def hook_fn(module, inp, out):
        feature_store.append(out.detach())

    handle = model.avgpool.register_forward_hook(hook_fn)

    transform = get_transforms("test")

    for domain in ["real", "sketch"]:
        items = sampled_images[domain]
        # Process in batches
        paths = [p for p, l in items]
        labels = [l for p, l in items]

        for start in range(0, len(paths), batch_size):
            end = min(start + batch_size, len(paths))
            batch_tensors = []
            for p in paths[start:end]:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))
            batch = torch.stack(batch_tensors).to(device)

            feature_store.clear()
            with torch.no_grad():
                model(batch)

            feats = feature_store[0].squeeze(-1).squeeze(-1)  # (batch, 2048)
            features_list.append(feats.cpu().numpy())
            labels_list.extend(labels[start:end])
            domains_list.extend([domain] * (end - start))

    handle.remove()

    features = np.concatenate(features_list, axis=0)
    labels_arr = np.array(labels_list)
    domains_arr = np.array(domains_list)
    print(f"    Extracted features: {features.shape}")
    return features, labels_arr, domains_arr


def plot_representations(features, labels, domains, class_names, output_dir):
    """Generate UMAP and t-SNE scatter plots."""
    print("\n  Generating representation plots ...")
    os.makedirs(os.path.join(output_dir, "representations"), exist_ok=True)

    num_classes = len(class_names)

    # UMAP
    print("    Running UMAP ...")
    umap_model = UMAP(n_components=2, random_state=42)
    umap_emb = umap_model.fit_transform(features)

    # t-SNE
    print("    Running t-SNE ...")
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_emb = tsne_model.fit_transform(features)

    # Color maps
    class_cmap = plt.cm.get_cmap("tab20", num_classes)
    domain_colors = {"real": "tab:blue", "sketch": "tab:orange"}

    for emb, name in [(umap_emb, "umap"), (tsne_emb, "tsne")]:
        # By class
        fig, ax = plt.subplots(figsize=(14, 10))
        for c in range(num_classes):
            mask = labels == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                       color=class_cmap(c), label=class_names[c])
        ax.set_title(f"{name.upper()} by Class")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=5,
                  ncol=2, markerscale=2)
        plt.tight_layout()
        path = os.path.join(output_dir, "representations",
                            f"{name}_by_class.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {path}")

        # By domain
        fig, ax = plt.subplots(figsize=(14, 10))
        for d in ["real", "sketch"]:
            mask = domains == d
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                       color=domain_colors[d], label=d)
        ax.set_title(f"{name.upper()} by Domain")
        ax.legend(fontsize=12, markerscale=3)
        plt.tight_layout()
        path = os.path.join(output_dir, "representations",
                            f"{name}_by_domain.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Config banner
    print("=" * 60)
    print("  XAI Explainability Pipeline")
    print("=" * 60)
    print(f"  Config:            {args.config}")
    print(f"  Checkpoint:        {args.checkpoint}")
    print(f"  Data root:         {args.data_root}")
    print(f"  Num samples/class: {args.num_samples}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  IG steps:          {args.ig_steps}")
    print(f"  LIME samples:      {args.lime_samples}")
    print(f"  Perturbations:     {args.num_perturbations}")
    print(f"  Num workers:       {args.num_workers}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:            {device}")
    print("=" * 60)

    # Load model
    print("\nStep 1: Loading model ...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    train_args = checkpoint.get("args", {})
    train_domain = train_args.get("domain", "unknown")

    model = models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Trained on: {train_domain}, classes: {num_classes}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}, "
          f"Best val F1: {checkpoint.get('best_val_f1', 'N/A')}")

    # Create output directory tree
    print("\nStep 2: Creating output directories ...")
    for subdir in ["attribution_maps", "stability", "faithfulness",
                    "cross_domain", "representations"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Sample images
    print("\nStep 3: Sampling images ...")
    splits_dir = os.path.join(args.data_root, "splits")
    sampled_images = {}
    for domain in ["real", "sketch"]:
        split_path = os.path.join(splits_dir, f"{domain}_test.json")
        all_samples = load_split(split_path)
        selected = sample_images(all_samples, args.num_samples)
        sampled_images[domain] = selected
        print(f"  {domain}: {len(selected)} images sampled")

    # Generate + cache all attribution maps
    print("\nStep 4: Generating attribution maps ...")
    cache, image_info = generate_all_attributions(
        model, sampled_images, class_names, device, args, args.output_dir
    )

    # Evaluate stability
    print("\nStep 5: Evaluating stability ...")
    evaluate_stability(model, sampled_images, cache, image_info, device, args,
                       args.output_dir)

    # Evaluate faithfulness
    print("\nStep 6: Evaluating faithfulness ...")
    evaluate_faithfulness(model, sampled_images, cache, image_info, device,
                          args.output_dir)

    # Evaluate cross-domain consistency
    print("\nStep 7: Evaluating cross-domain consistency ...")
    evaluate_cross_domain(cache, sampled_images, class_names, args.output_dir)

    # Extract features + plot representations
    print("\nStep 8: Representation analysis ...")
    features, labels_arr, domains_arr = extract_features(
        model, sampled_images, device, batch_size=args.batch_size
    )
    plot_representations(features, labels_arr, domains_arr, class_names,
                         args.output_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("  XAI Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {args.output_dir}/")
    print(f"  Attribution maps: attribution_maps/[method]/[domain]/")
    print(f"  Stability:        stability/stability_scores.csv")
    print(f"  Faithfulness:     faithfulness/deletion_scores.csv")
    print(f"                    faithfulness/insertion_scores.csv")
    print(f"  Cross-domain:     cross_domain/cross_domain_consistency.csv")
    print(f"  Representations:  representations/*.png")
    print(f"{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
