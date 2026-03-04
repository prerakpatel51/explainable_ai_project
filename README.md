# Explainable AI on DomainNet: Cross-Domain Attribution Analysis

A deep learning + explainability research project that trains **ResNet-152** classifiers on the [DomainNet](http://ai.bu.edu/M3SDA/) benchmark and rigorously evaluates **four XAI attribution methods** across two visual domains — photographic (*real*) and hand-drawn (*sketch*).

The central question this project answers:

> *When a neural network is trained on one visual style, do its explanations remain consistent, faithful, and stable — even on images it has never seen before?*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset & Classes](#dataset--classes)
- [Project Structure](#project-structure)
- [Pipeline at a Glance](#pipeline-at-a-glance)
- [Step 1 — Data Preparation (`prepare_data.py`)](#step-1--data-preparation-prepare_datapy)
- [Step 2 — Model Training (`train.py`)](#step-2--model-training-trainpy)
- [Step 3 — Evaluation (`evaluate.py`)](#step-3--evaluation-evaluatepy)
- [Step 4 — XAI Explainability (`explain.py`)](#step-4--xai-explainability-explainpy)
  - [Method 1: Grad-CAM](#method-1-grad-cam)
  - [Method 2: Grad-CAM++](#method-2-grad-cam)
  - [Method 3: Integrated Gradients](#method-3-integrated-gradients)
  - [Method 4: LIME](#method-4-lime)
  - [Evaluation Axis 1: Stability](#evaluation-axis-1-stability)
  - [Evaluation Axis 2: Faithfulness](#evaluation-axis-2-faithfulness)
  - [Evaluation Axis 3: Cross-Domain Consistency](#evaluation-axis-3-cross-domain-consistency)
  - [Evaluation Axis 4: Representation Behavior](#evaluation-axis-4-representation-behavior)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [Running on SLURM (`run_real.sh` / `run_sketch.sh`)](#running-on-slurm-run_realsh--run_sketchsh)
- [Output Structure](#output-structure)
- [Installation](#installation)

---

## Project Overview

**DomainNet** contains images of the same objects across 6 visual domains. This project focuses on two of them:

| Domain   | Description                             | Visual Style         |
|----------|-----------------------------------------|----------------------|
| `real`   | Photographs from the internet           | Photo-realistic      |
| `sketch` | Hand-drawn pencil / line-art sketches   | Abstract / structural |

**81 classes** were selected from the original 345, removing 19 underperforming categories identified in prior evaluation rounds.

Two independent models are trained — one per domain — and then cross-evaluated and explained. The XAI pipeline applies four attribution methods and measures each along four quantitative axes, producing a complete picture of *how* and *how reliably* the models reason.

---

## Dataset & Classes

The 81 retained classes span 10 semantic groups:

| Category              | Count | Examples                                       |
|-----------------------|-------|------------------------------------------------|
| Animals / Nature      | 21    | duck, lion, spider, swan, tiger, palm_tree     |
| Objects / Tools       | 27    | axe, laptop, saxophone, screwdriver, sword     |
| Vehicles / Transport  | 5     | airplane, bicycle, firetruck, sailboat         |
| Body / Human          | 2     | skull, beard                                   |
| Places / Architecture | 5     | barn, lighthouse, skyscraper, stairs           |
| Sports / Activity     | 4     | baseball, basketball, soccer_ball              |
| Symbols / Shapes      | 3     | circle, triangle, spreadsheet                  |
| Landmarks             | 2     | The Eiffel Tower, The Mona Lisa                |
| Food                  | 3     | banana, wine_glass, crown                      |
| Other                 | 8     | light_bulb, telephone, trumpet, toaster        |

> **Excluded classes** (19 total): skateboard, smiley_face, sleeping_bag, see_saw, golf_club, arm, snorkel, stop_sign, table, camouflage, crayon, rain, knee, fork, sun, tooth, picture_frame, streetlight, ear — removed due to weak inter-domain alignment or insufficient samples.

---

## Project Structure

```
xai_project_2/
│
├── config.yaml          # Central configuration for all scripts
├── prepare_data.py      # Data splitting, balancing, and weight computation
├── dataset.py           # PyTorch Dataset class + image transforms
├── train.py             # ResNet-152 training with mixup, cosine LR, early stopping
├── evaluate.py          # Test-set evaluation + confusion matrix + training curves
├── explain.py           # Full XAI pipeline: 4 methods × 4 evaluation axes
│
├── run_real.sh          # SLURM job: full real-domain pipeline (train→eval→explain)
├── run_sketch.sh        # SLURM job: full sketch-domain pipeline
├── requirements.txt     # Python dependencies
│
├── splits/              # Auto-generated data splits (JSON)
│   ├── class_names.json
│   ├── real_train.json / real_val.json / real_test.json
│   ├── sketch_train.json / sketch_val.json / sketch_test.json
│   ├── real_class_weights.json / sketch_class_weights.json
│   └── balancing_report.json
│
├── output/
│   ├── models/
│   │   ├── real/        # best_model.pt + checkpoints/ + training_log.csv
│   │   └── sketch/
│   ├── evaluation/      # Per-domain evaluation results, plots, confusion matrices
│   └── xai/
│       ├── real_model/  # Attribution maps + stability/faithfulness/cross-domain CSVs
│       └── sketch_model/
│
└── logs/                # SLURM stdout/stderr logs
```

---

## Pipeline at a Glance

```
Raw DomainNet Images
        │
        ▼
 prepare_data.py  ──►  Stratified 80/10/10 splits
                        Cross-domain oversampling balance
                        Inverse-frequency class weights
        │
        ▼
   train.py       ──►  ResNet-152 (ImageNet pretrained)
                        Mixup augmentation
                        Cosine LR + linear warmup
                        Weighted CrossEntropy + label smoothing
                        Early stopping on val loss
        │
        ▼
  evaluate.py     ──►  Accuracy / F1 / Precision / Recall
                        Confusion matrix
                        Training curves
                        In-domain AND cross-domain evaluation
        │
        ▼
  explain.py      ──►  Grad-CAM  |  Grad-CAM++
                        Integrated Gradients  |  LIME
                        ↓
                   Stability (SSIM under perturbation)
                   Faithfulness (deletion / insertion AUC)
                   Cross-domain consistency (cosine similarity)
                   Representation behavior (UMAP / t-SNE)
```

---

## Step 1 — Data Preparation (`prepare_data.py`)

**Goal:** Build clean, balanced, reproducible train/val/test splits from raw DomainNet folders.

### What it does

1. **Scans** the `real/` and `sketch/` directories for images matching the 81 selected classes.
2. **Stratified splitting** — each class maintains its proportion across train / val / test:
   - 80% training, 10% validation, 10% test
   - Uses `sklearn.model_selection.train_test_split` with `stratify=labels`
3. **Cross-domain oversampling balance** — since `real` and `sketch` have different per-class counts, the training sets are balanced:
   - For each class: `target_count = max(real_count, sketch_count)`
   - The smaller domain is oversampled by randomly duplicating entries
   - Val / test splits are **never** modified — only training is balanced
   - Training augmentations (random crop, flip, jitter) make duplicated images look different each epoch
4. **Class weight computation** — inverse-frequency weights for the loss function:
   ```
   weight_c = total_samples / (num_classes × count_c)
   ```
5. **Saves** all splits as JSON files (`[path, label]` pairs) and a `balancing_report.json` summarizing per-class oversampling.

### Key design decisions

- Val/test are held out before balancing — this prevents data leakage and ensures unbiased evaluation.
- Labels are assigned by **sorted** class name, making the mapping deterministic and reproducible.
- A fixed `random_state=42` is used throughout for reproducibility.

---

## Step 2 — Model Training (`train.py`)

**Goal:** Train a high-accuracy classifier on either the `real` or `sketch` domain using transfer learning from ImageNet.

### Architecture

| Component    | Choice                                              |
|--------------|-----------------------------------------------------|
| Backbone     | **ResNet-152** (pretrained on ImageNet-1K V2)       |
| Final layer  | `Linear(2048 → 81)` replacing the original fc      |
| Multi-GPU    | `nn.DataParallel` for automatic data parallelism    |

### Training Techniques

#### Frozen warm-up phase (first 5 epochs)
Early layers (`conv1`, `bn1`, `layer1`, `layer2`) are frozen initially so the new classification head can stabilize before updating the pretrained weights:
```
Frozen:    conv1, bn1, layer1, layer2  (first 5 epochs)
Trainable: layer3, layer4, fc         (all epochs)
```
After epoch 5, all layers are unfrozen and the full model fine-tunes.

#### Differential learning rates
The pretrained backbone and the new head are optimized at different rates to avoid catastrophic forgetting:
- Backbone parameters: `lr × 0.1`
- FC head parameters: `lr` (full learning rate)

#### Cosine annealing with linear warmup
```
Epochs 0–4:  Linear warmup   (lr ramps from 0 → lr)
Epochs 5–N:  Cosine decay    (lr decays from lr → 0.01 × lr)
```

#### Mixup augmentation
At each training step, two random samples are blended:
```python
mixed_x = λ × x_i + (1 − λ) × x_j    where λ ~ Beta(0.2, 0.2)
loss = λ × CE(pred, y_i) + (1 − λ) × CE(pred, y_j)
```
This regularizes the model and improves cross-domain generalization.

#### Weighted CrossEntropy + label smoothing
- Inverse-frequency class weights from `prepare_data.py` counter class imbalance.
- `label_smoothing=0.1` prevents overconfident predictions.

#### Image augmentation pipeline (training)
```
RandomResizedCrop(224, scale=0.7–1.0)
RandomHorizontalFlip
RandomRotation(±15°)
RandomAffine(translate=10%, scale=0.9–1.1)
ColorJitter(brightness, contrast, saturation=0.3, hue=0.1)
RandomGrayscale(p=0.1)
ToTensor → Normalize(ImageNet mean/std)
RandomErasing(p=0.2)
```

#### Early stopping & checkpointing
- Best model saved by **macro F1** on validation set.
- Checkpoint saved every epoch to allow training resumption.
- Early stopping triggers after `patience=10` epochs with no validation loss improvement.

---

## Step 3 — Evaluation (`evaluate.py`)

**Goal:** Comprehensively evaluate a trained model — both in-domain and cross-domain — and visualize results.

### Metrics computed

| Metric              | Description                                            |
|---------------------|--------------------------------------------------------|
| Top-1 Accuracy      | Overall classification accuracy                        |
| Macro F1            | F1 averaged equally across all 81 classes              |
| Weighted F1         | F1 weighted by class support                           |
| Macro Precision     | Mean precision across classes                          |
| Macro Recall        | Mean recall across classes                             |
| Per-class breakdown | Accuracy, F1, precision, recall for each of 81 classes |

### Outputs generated

- **Confusion matrix** heatmap (81×81) — highlights systematic class confusions.
- **Training curves** — loss, accuracy, F1, precision, recall, and learning rate over epochs.
- **Per-class bar charts** — sorted by performance to identify hard and easy classes.
- **Random sample predictions** — a grid of test images with predicted vs. true labels.
- All numeric results exported to CSV for downstream analysis.

### Evaluation scenarios

Each trained model is evaluated on **two** test sets:

| Trained on | Evaluated on | Type           |
|------------|--------------|----------------|
| `real`     | `real`       | In-domain      |
| `real`     | `sketch`     | Cross-domain   |
| `sketch`   | `sketch`     | In-domain      |
| `sketch`   | `real`       | Cross-domain   |

Cross-domain evaluation reveals how well the model generalizes to a different visual style without any retraining.

---

## Step 4 — XAI Explainability (`explain.py`)

**Goal:** Apply four attribution methods to the trained model and rigorously measure explanation quality along four axes.

Attribution methods answer: *"Which pixels drove the model to predict this class?"*

---

### Method 1: Grad-CAM

**Gradient-weighted Class Activation Mapping**

Grad-CAM uses the gradients of the target class score flowing into the final convolutional layer (`layer4`) to produce a coarse spatial heatmap.

**How it works:**
1. Forward pass → record `layer4` feature maps **A** ∈ ℝ^(2048×7×7)
2. Backward pass → record gradients **∂y_c / ∂A** ∈ ℝ^(2048×7×7)
3. Global average pool the gradients over spatial dimensions → weights **α_k** ∈ ℝ^2048
4. Weighted sum of feature maps + ReLU:
   ```
   L_GradCAM = ReLU( Σ_k α_k × A_k )
   ```
5. Bilinear upsample to 224×224

**Strengths:** Fast, class-discriminative, highlights which spatial regions matter.
**Limitations:** Coarse resolution (7×7 base), can miss fine-grained details.

---

### Method 2: Grad-CAM++

An improved version of Grad-CAM with better multi-instance localization.

**How it works:**
Instead of simple global-average-pooled gradients, Grad-CAM++ computes pixel-wise importance weights using second and third-order gradients:

```
α_k^c = Σ_{i,j} (∂²y_c / ∂A_k²) / [ 2(∂²y_c / ∂A_k²) + Σ(A_k × ∂³y_c / ∂A_k³) ]
```

Practically: the weights `α` account for how much each activation contributes to the positive gradient of the class score, giving more precise localization.

**Strengths:** Better at capturing multiple instances of the same class, sharper boundaries.
**Difference from Grad-CAM:** More numerically stable weights that better handle cases where a class appears multiple times.

---

### Method 3: Integrated Gradients

A gradient-based attribution method with a formal axiomatic foundation (satisfies *completeness* and *sensitivity* axioms).

**How it works:**
1. Define a **baseline** (black image, zero tensor in normalized space).
2. Interpolate `ig_steps=100` images between baseline and input: `x_α = baseline + α × (input − baseline)`
3. Compute gradient of target class score w.r.t. each interpolated image.
4. Average gradients using the trapezoidal rule (numerical integration):
   ```
   IG(x) = (x − x') × ∫₀¹ (∂F / ∂x)|_{x=x'+α(x−x')} dα
   ```
5. Sum absolute attributions across color channels → per-pixel importance.

**Strengths:** Theoretically grounded, pixel-level precision (224×224 native resolution), no spatial pooling.
**Limitations:** Slower (requires `ig_steps` forward+backward passes), sensitive to baseline choice.

---

### Method 4: LIME

**Local Interpretable Model-agnostic Explanations**

LIME treats the model as a black box and explains predictions by fitting a simple linear model in the neighborhood of each input.

**How it works:**
1. **Segment** the image into superpixels using quickshift.
2. Generate `lime_samples=3000` perturbed versions by randomly turning superpixels on/off (hidden with a constant color).
3. Get model predictions on all perturbed images.
4. Fit a **weighted linear regression** in the superpixel space, with samples weighted by their similarity to the original image.
5. The regression coefficients become the per-superpixel importance scores.

**Strengths:** Model-agnostic (works on any classifier), produces human-interpretable superpixel regions.
**Limitations:** Slowest method, inherently stochastic (though `random_state=42` is fixed), coarser than pixel-level methods.

---

### Evaluation Axis 1: Stability

**Question:** Do attribution maps stay consistent when the input is slightly perturbed?

**Method:**
- For each image, generate `num_perturbations=5` variants using either:
  - **Gaussian noise** (σ=0.05 in normalized space)
  - **Small rotation** (±5°, randomly chosen)
- Compute attribution maps for each perturbed version.
- Compare original and perturbed maps using **SSIM** (Structural Similarity Index).

**Interpretation:**
- SSIM ∈ [−1, 1], higher = more stable
- A good explainer should produce nearly identical heatmaps for semantically identical inputs
- Low SSIM suggests the method is sensitive to input noise → unreliable for decision support

**Output:** `stability/stability_scores.csv` — mean and std SSIM per method per domain.

---

### Evaluation Axis 2: Faithfulness

**Question:** Do the highlighted pixels actually matter to the model's prediction?

Uses the **deletion** and **insertion** protocols:

#### Deletion test
Progressively mask the *most important* pixels (zeroed out in normalized space) from 10% to 90%.
- Record class probability at each masking level.
- Compute AUC of the probability curve.
- **Lower AUC = more faithful** (probability drops steeply when important pixels are removed).

#### Insertion test
Start from a blank image and progressively *reveal* the most important pixels from 10% to 90%.
- Record class probability at each reveal level.
- Compute AUC of the probability curve.
- **Higher AUC = more faithful** (probability rises quickly as important pixels are revealed).

**Output:** `faithfulness/deletion_scores.csv` and `faithfulness/insertion_scores.csv`.

---

### Evaluation Axis 3: Cross-Domain Consistency

**Question:** Does the model focus on the same semantic features when it sees a photograph vs. a sketch of the same class?

**Method:**
For each class:
1. Compute average (prototype) attribution map across all `real` samples of that class.
2. Compute average attribution map across all `sketch` samples of that class.
3. Measure **cosine similarity** between the two flattened prototype vectors.

**Interpretation:**
- Cosine similarity ∈ [−1, 1], higher = more consistent across domains
- High consistency means the model attends to the same structural features regardless of visual style (e.g., always focuses on the wings of an airplane)
- Low consistency suggests domain-specific shortcuts or texture bias

**Output:** `cross_domain/cross_domain_consistency.csv` — per-class and per-method similarity scores.

---

### Evaluation Axis 4: Representation Behavior

**Question:** How does the model organize visual concepts in its internal feature space — and does the real/sketch domain boundary matter?

**Method:**
1. Extract 2048-dimensional feature vectors from `model.avgpool` for all sampled images using a forward-hook.
2. Reduce to 2D using:
   - **UMAP** (preserves global + local structure, faster)
   - **t-SNE** (preserves local neighborhood structure)
3. Visualize scatter plots colored by:
   - **Class label** — are semantically similar classes clustered together?
   - **Domain** — does the model create a real/sketch separation or domain-invariant embeddings?

**Output:** `representations/umap_by_class.png`, `representations/umap_by_domain.png`, `representations/tsne_by_class.png`, `representations/tsne_by_domain.png`.

---

## Configuration (`config.yaml`)

All scripts read defaults from `config.yaml`. Any setting can be overridden via CLI flags.

```yaml
data:
  data_root: /path/to/xai_project_2
  num_workers: 8
  classes: [...]        # 81 selected class names

training:
  domain: real          # 'real' or 'sketch'
  epochs: 100
  batch_size: 256
  lr: 0.001
  patience: 10
  output_dir: output
  resume: null          # path to checkpoint to resume from

evaluation:
  checkpoint: output/models/real/best_model.pt
  test_domain: real
  batch_size: 64
  output_dir: output/evaluation

explain:
  checkpoint: output/models/real/best_model.pt
  num_samples: 5        # images per class per domain
  output_dir: output/xai/real_model
  batch_size: 32
  ig_steps: 100         # interpolation steps for Integrated Gradients
  lime_samples: 3000    # perturbation samples for LIME
  num_perturbations: 5  # stability test perturbations per image
```

---

## Running on SLURM (`run_real.sh` / `run_sketch.sh`)

The project ships with full SLURM batch scripts for HPC clusters. Each script runs the **complete pipeline** end-to-end:

```
prepare_data → train → eval (in-domain) → eval (cross-domain) → explain
```

**SLURM resource request per job:**
```
partition:    h200
nodes:        1
cpus:         16
gpus:         8 × H200
memory:       200 GB
```

**Submit the jobs:**
```bash
sbatch run_real.sh     # Real domain pipeline
sbatch run_sketch.sh   # Sketch domain pipeline
```

**Override any setting without editing the script:**
```bash
python train.py --domain sketch --epochs 50 --lr 0.0005
python evaluate.py --checkpoint output/models/real/best_model.pt --test_domain sketch
python explain.py --num_samples 10 --ig_steps 200 --lime_samples 5000
```

---

## Output Structure

```
output/
├── models/
│   ├── real/
│   │   ├── best_model.pt            # Best checkpoint (by macro F1)
│   │   ├── training_log.csv         # Per-epoch metrics
│   │   └── checkpoints/
│   │       ├── epoch_01.pt
│   │       ├── epoch_02.pt
│   │       └── ...
│   └── sketch/
│       └── (same structure)
│
├── evaluation/
│   ├── real_model_on_real/
│   │   ├── metrics.csv
│   │   ├── confusion_matrix.png
│   │   ├── training_curves.png
│   │   ├── per_class_accuracy.png
│   │   └── sample_predictions.png
│   ├── real_model_on_sketch/
│   ├── sketch_model_on_sketch/
│   └── sketch_model_on_real/
│
└── xai/
    ├── real_model/
    │   ├── attribution_maps/
    │   │   ├── gradcam/real/          # 3-panel PNGs: original|heatmap|overlay
    │   │   ├── gradcam/sketch/
    │   │   ├── gradcam_pp/real/
    │   │   ├── gradcam_pp/sketch/
    │   │   ├── integrated_gradients/real/
    │   │   ├── integrated_gradients/sketch/
    │   │   ├── lime/real/
    │   │   └── lime/sketch/
    │   ├── stability/
    │   │   └── stability_scores.csv
    │   ├── faithfulness/
    │   │   ├── deletion_scores.csv
    │   │   └── insertion_scores.csv
    │   ├── cross_domain/
    │   │   └── cross_domain_consistency.csv
    │   └── representations/
    │       ├── umap_by_class.png
    │       ├── umap_by_domain.png
    │       ├── tsne_by_class.png
    │       └── tsne_by_domain.png
    └── sketch_model/
        └── (same structure)
```

---

## Installation

```bash
# Clone the repo
git clone git@github.com:prerakpatel51/xai_project.git
cd xai_project

# Create and activate environment
conda create -n xai python=3.10 -y
conda activate xai

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
```
torch>=2.0          torchvision>=0.15
numpy>=1.24         matplotlib>=3.7
scikit-learn>=1.3   scikit-image>=0.21
scipy>=1.11         lime>=0.2
umap-learn>=0.5     Pillow>=9.0
```

**Data setup:** Place the DomainNet `real/` and `sketch/` folders inside the project root, then run:
```bash
python prepare_data.py   # generates all splits automatically
```

---

*Built as part of an XAI research project studying attribution method reliability across visual domains.*
