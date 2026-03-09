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

**81 classes** were selected from the original 345.
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

Deep neural networks achieve impressive accuracy, but they function as "black boxes" — they don't explain *why* they make a prediction. **Attribution methods** (also called *saliency methods*) address this by assigning an importance score to every spatial location in the input image. The result is a **heatmap** that highlights the pixels or regions most influential to the model's decision. For example, if the model predicts "airplane", the heatmap should highlight the wings and fuselage rather than the background sky.

---

### Method 1: Grad-CAM

**Gradient-weighted Class Activation Mapping**

Grad-CAM is one of the most widely used XAI methods in computer vision. It works by looking at the *last convolutional layer* of the network — the layer that still retains spatial information (it knows *where* things are) while also encoding high-level semantic concepts (it knows *what* things look like).

#### Background: Why the last convolutional layer?

A CNN like ResNet-152 processes an image through progressively more abstract layers:
- **Early layers** (conv1, layer1): detect edges, corners, colour gradients
- **Middle layers** (layer2, layer3): combine these into textures, parts, object fragments
- **Late layers** (layer4): detect high-level concepts — "this region contains a wheel", "this region has a face"

After layer4, the spatial information is collapsed by global average pooling into a single vector, and a fully connected layer produces class scores. Grad-CAM taps in *just before* this collapse, capturing both what the model has learned and where each concept appears in the image.

#### Step-by-step computation

1. **Forward pass:** The input image (224×224×3) is fed through the network. A PyTorch *forward hook* on `layer4` records the output **feature maps** A ∈ ℝ^(2048×7×7). Think of this as 2,048 small 7×7 images, each encoding a different visual pattern. For example, one channel might activate wherever it sees "fur texture", another wherever it sees "circular shape".

2. **Backward pass:** We select the class to explain (say, "airplane") and compute the gradient of its score y^c with respect to the feature maps. These gradients tell us how a small change at each spatial location in each channel would affect the airplane score. Large positive gradients mean "helpful for predicting airplane."

3. **Global average pooling of gradients:** Average each channel's gradients over its 7×7 grid to get a single importance weight per channel:
   ```
   α_k = (1/Z) × Σ_{i,j} ∂y_c / ∂A_{k,i,j}
   ```
   A high α_k means channel k's feature pattern is important for class c. If the "wheel" channel has high weight for "bicycle", it means the model considers wheels strong evidence for bicycles.

4. **Weighted combination + ReLU:**
   ```
   L_GradCAM = ReLU( Σ_k α_k × A_k )
   ```
   The ReLU clips negative values to zero — negative values correspond to features of *other* classes. When explaining "airplane", we don't want to highlight regions that look like "bicycle".

5. **Upsampling:** The 7×7 heatmap is bilinearly interpolated to 224×224 to overlay on the original image.

**Strengths:** Fast (one forward + backward pass), class-discriminative (different classes → different heatmaps), intuitive output.

**Limitations:** Coarse resolution (7×7 base) — shows *which region* matters but not *which specific pixels*. Tends to highlight only the most discriminative part of an object (e.g., the face but not the body). No formal mathematical guarantees.

---

### Method 2: Grad-CAM++

An improved version of Grad-CAM that addresses two practical problems: (1) Grad-CAM often highlights only the most discriminative part of an object (face of a dog but not the body), and (2) it struggles when multiple instances of the same class appear (a flock of birds).

#### The core improvement

The only change from Grad-CAM is *how the channel importance weights are computed*. Instead of simple global average pooling of gradients, Grad-CAM++ uses **higher-order derivatives** (second and third derivatives of the class score) to compute pixel-wise weights:

```
α_k^c = Σ_{i,j} (∂²y_c / ∂A_k²) / [ 2(∂²y_c / ∂A_k²) + Σ(A_k × ∂³y_c / ∂A_k³) + ε ]
```

#### Intuition

Grad-CAM asks: "On average across all positions, does this channel help predict dog?" Grad-CAM++ asks: "At *each specific position*, how much does this pixel contribute to the positive evidence for dog?" By considering second-order derivatives (curvature), the method captures not just *whether* the gradient is positive but *how quickly it is changing*. The practical effect is that Grad-CAM++ distributes attention more evenly across the entire object:

- Grad-CAM on a dog → highlights only the face
- Grad-CAM++ on the same dog → highlights face, body, legs, and tail

**Strengths:** Better full-object coverage, more numerically stable weights, better multi-instance localization, same computational cost as Grad-CAM (one forward + backward pass).

**Limitations:** Still limited to 7×7 resolution, higher-order gradients can be unstable for some architectures (works well for ResNets), still a heuristic without formal guarantees.

---

### Method 3: Integrated Gradients

A gradient-based attribution method with a formal axiomatic foundation. Unlike CAM-based methods that operate on intermediate feature maps, Integrated Gradients computes attributions directly at the **input pixel level**, producing 224×224 resolution.

#### The problem with simple gradients

You might think: "Just compute the gradient of the class score with respect to each pixel — large gradients mean important pixels." This is called the *vanilla gradient* approach, and it has a critical flaw: **gradient saturation**. Neural networks with ReLU activations have flat regions where the gradient is zero even though the feature is important. A pixel might be crucial for recognition, but if the network is saturated, the gradient is zero and the pixel gets no attribution.

#### Axiomatic foundation

Integrated Gradients satisfies two formal axioms that any "fair" attribution method should obey:

- **Sensitivity:** If changing a single pixel changes the output, that pixel *must* receive a non-zero attribution.
- **Completeness:** The sum of all pixel attributions exactly equals the difference between the model's output for the input and the baseline: `Σ IG_i(x) = F(x) − F(x')`. No importance is lost or fabricated.

#### Step-by-step computation

1. **Choose a baseline x':** A black image (zero tensor after normalization) representing "absence of information" — what the model sees when "nothing" is present.

2. **Create interpolated images:** Generate n=100 intermediate images that gradually blend from baseline to input:
   ```
   x_α = x' + α × (x − x')   for α = 0, 1/100, 2/100, ..., 1
   ```
   Imagine slowly "fading in" the photograph from black.

3. **Compute gradients at each step:** For each interpolated image, compute the gradient of the class score. This tells us, at each point along the fade-in, which pixels are currently contributing most.

4. **Integrate the gradients:**
   ```
   IG_i(x) = (x_i − x'_i) × ∫₀¹ (∂F / ∂x_i)|_{x=x'+α(x−x')} dα
   ```
   In practice, we use the trapezoidal rule to numerically approximate the integral.

5. **Aggregate across channels:** Sum absolute attributions over the 3 RGB channels → single per-pixel importance score.

#### Why integration helps

Consider a pixel vital for recognizing a tiger's stripe. At the baseline (black), this pixel contributes nothing. As we fade in, there's a critical moment when this pixel "activates" the stripe detector — the gradient spikes. By integrating over the entire path, IG captures this spike even if the gradient is zero at the endpoints. Simple gradient methods would miss it entirely.

**Strengths:** Pixel-level resolution (224×224), mathematically principled (axiomatic guarantees), architecture-agnostic (works for any differentiable model).

**Limitations:** Slow (101 forward+backward passes per image, ~100× slower than Grad-CAM), noisier heatmaps (full resolution with no smoothing), baseline-dependent (black vs. white vs. random can change results).

---

### Method 4: LIME

**Local Interpretable Model-agnostic Explanations**

LIME takes a fundamentally different philosophy: it does *not* look inside the network at all. It treats the neural network as a complete **black box** and explains predictions by observing how the model's output changes when parts of the input are hidden.

#### The core idea

Imagine covering parts of a photograph with grey patches and seeing how the prediction changes. If covering the dog's face causes "dog" probability to drop sharply, the face is clearly important. LIME automates this by: (1) dividing the image into meaningful regions (superpixels), (2) systematically hiding different combinations, and (3) fitting a simple model (linear regression) to learn which regions matter most.

#### Step-by-step computation

1. **Segment into superpixels:** Using quickshift segmentation, the image is divided into ~150 superpixels — contiguous regions of visually similar pixels. Each might correspond to a patch of sky, a section of a wheel, or part of a face.

2. **Generate 3,000 perturbations:** For each perturbation, a random subset of superpixels is "turned off" (replaced with grey, pixel value 128). Each perturbation is encoded as a binary vector z ∈ {0,1}^150 indicating which superpixels are visible (1) or hidden (0).

3. **Query the model:** All 3,000 perturbed images are passed through the network to get predicted class probabilities. This is the most expensive step.

4. **Fit weighted linear regression:** In the binary superpixel space, fit a linear model:
   ```
   g(z) = w_0 + Σ_j w_j × z_j
   ```
   Samples more similar to the original (more superpixels visible) receive higher weights. The coefficients w_j become per-superpixel importance scores.

5. **Interpret coefficients:** Positive w_j means superpixel j *supports* the prediction — hiding it reduces confidence. Negative w_j means it *hurts* the prediction. We keep only positive contributions and normalize.

#### Why model-agnostic matters

LIME doesn't need access to weights, gradients, or architecture — just the ability to query the model. This makes it applicable to proprietary APIs (model not downloadable), non-differentiable models (random forests, ensembles), and multi-model pipelines.

**Strengths:** Model-agnostic (works with any classifier), human-interpretable (superpixel regions are visually meaningful), captures feature interactions implicitly.

**Limitations:** Slowest method (3,000 forward passes per image), inherently stochastic (different perturbations → slightly different explanations, though we fix `random_state=42`), superpixel-dependent (poor segmentation → poor explanations), local fidelity only.

---

### Method Comparison

| Property | Grad-CAM | Grad-CAM++ | Int. Grad. | LIME |
|----------|----------|------------|------------|------|
| Resolution | 7×7 | 7×7 | 224×224 | Superpixel |
| Model-agnostic | No | No | No | Yes |
| Gradient-based | Yes | Yes | Yes | No |
| Axiomatic | No | No | Yes | No |
| Speed | Fast | Fast | Moderate | Slow |
| Fwd+bwd passes | 1 | 1 | 101 | 3,000 (fwd) |
| Deterministic | Yes | Yes | Yes | No |

---

### Evaluation Axis 1: Stability

**Question:** Do attribution maps stay consistent when the input is slightly perturbed?

A reliable explanation should not change drastically when the input changes imperceptibly. If adding invisible noise causes the heatmap to shift from the dog's face to the background, the explanation is unreliable.

**Protocol:**
- For each image, generate 5 perturbations (Gaussian noise σ=0.05 or small rotation ±5°)
- Compute attribution maps for each perturbation
- Measure **SSIM** (Structural Similarity Index) between original and perturbed maps
- SSIM ∈ [−1, 1], higher = more stable

**Output:** `stability/stability_scores.csv`

---

### Evaluation Axis 2: Faithfulness

**Question:** Do the highlighted pixels actually matter to the model's decision?

A heatmap might highlight the right region by coincidence. Faithfulness tests whether removing (or adding) the highlighted pixels actually changes the prediction.

**Deletion protocol:** Progressively remove the most "important" pixels (10% → 90%) and record confidence drop. A faithful explanation causes rapid confidence drop → **higher deletion AUC = more faithful**.

**Insertion protocol:** Start from blank, progressively reveal the most "important" pixels (10% → 90%) and record confidence rise. A faithful explanation causes rapid confidence rise → **higher insertion AUC = more faithful**.

**Output:** `faithfulness/deletion_scores.csv` and `faithfulness/insertion_scores.csv`

---

### Evaluation Axis 3: Cross-Domain Consistency

**Question:** Does the model focus on the same features when viewing a photograph vs. a sketch of the same class?

**Protocol:**
1. For each class, compute average (prototype) attribution map across all real images
2. Compute average attribution map across all sketch images
3. Measure **cosine similarity** between the two prototypes
- Higher = more consistent features across visual styles
- Lower = domain-specific shortcuts or texture bias

**Output:** `cross_domain/cross_domain_consistency.csv`

---

### Evaluation Axis 4: Representation Behavior

**Question:** How does the model organize images in its internal feature space? Does it group by class (good) or by domain (problematic)?

**Protocol:**
1. Extract 2048-D feature vectors from penultimate layer (`model.avgpool`)
2. Project to 2D using **t-SNE** (local structure) and **UMAP** (global + local)
3. Colour by class (do classes cluster?) and by domain (do domains separate?)

**Output:** `representations/umap_by_class.png`, `umap_by_domain.png`, `tsne_by_class.png`, `tsne_by_domain.png`

---

### Implementation Details

All XAI methods and evaluation axes are implemented in `explain.py`. Here is how each component works under the hood:

#### Attribution Generation (`generate_all_attributions`)

For each of the 4 methods × 2 domains × 20 samples/class × 81 classes = 12,960 images:
1. The image is loaded and preprocessed using the same transforms as evaluation (Resize(256) → CenterCrop(224) → Normalize).
2. The attribution function is called, producing a 224×224 heatmap normalized to [0, 1].
3. A 3-panel visualization is saved as PNG: **original | heatmap (jet colormap) | overlay** (50% blend of original and heatmap).
4. All heatmaps are cached in memory (a dict keyed by `method → domain → index`) for reuse by the evaluation axes — this avoids recomputing attributions multiple times.

#### Grad-CAM / Grad-CAM++ Implementation

Both methods use PyTorch **forward hooks** and **backward hooks** on `model.layer4` (the final convolutional block):
```python
# Forward hook: captures feature maps A ∈ ℝ^(2048×7×7)
handle_fwd = model.layer4.register_forward_hook(lambda m, i, o: activations.append(o))
# Backward hook: captures gradients ∂y_c/∂A
handle_bwd = model.layer4.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
```
After the forward pass, the target class score is backpropagated, and the hooks capture both the feature maps and their gradients. Channel weights are computed (simple averaging for Grad-CAM, higher-order weighting for Grad-CAM++), and the weighted sum is ReLU'd and upsampled via `cv2.resize` to 224×224.

#### Integrated Gradients Implementation

Uses the path integral approach with `ig_steps=100` interpolation steps:
```python
# Create interpolated inputs from baseline (black) to actual input
scaled_inputs = [baseline + α/steps * (input − baseline) for α in range(steps+1)]
# Batch the forward+backward passes for efficiency
# Aggregate gradients using trapezoidal rule
# Multiply by (input − baseline) and sum across color channels
```
The implementation batches the interpolated images using `args.batch_size` to fit GPU memory, then aggregates the gradients.

#### LIME Implementation

Uses the `lime` Python package (`LimeImageExplainer`):
```python
explainer = LimeImageExplainer()
explanation = explainer.explain_instance(
    image_numpy,                    # uint8 RGB image
    batch_predict_fn,               # wraps model inference
    top_labels=1,
    num_samples=3000,               # perturbation count
    random_state=42                 # reproducibility
)
```
The `batch_predict_fn` handles preprocessing (ToTensor + Normalize) and batched GPU inference for the 3,000 perturbed images. Explanation weights for the top predicted label are extracted and mapped back to pixel space via the superpixel segmentation mask.

#### Stability Evaluation (`evaluate_stability`)

For each cached attribution map:
1. Load the original image tensor and apply a random perturbation via `perturb_tensor()`:
   - **Gaussian noise** (50% chance): adds `N(0, 0.05)` noise in normalized space
   - **Rotation** (50% chance): applies `torchvision.transforms.functional.rotate()` with angle ∈ [−5°, +5°]
2. Recompute the attribution map for the perturbed image using the same method.
3. Compute **SSIM** between the original and perturbed heatmaps using `skimage.metrics.structural_similarity`.
4. Repeat 5 times per image, aggregate mean and std per method per domain.

#### Faithfulness Evaluation (`evaluate_faithfulness`)

For each cached attribution map:
1. Sort all 224×224 = 50,176 pixel indices by attribution score (highest first).
2. **Deletion test:** At each percentage level (10%, 20%, ..., 90%), create a mask that zeros out the top-k most important pixels. Feed the masked image to the model and record the target class probability. Compute AUC over the 9 probability values using `np.trapz`.
3. **Insertion test:** Start from a zero tensor. At each level, reveal the top-k pixels from the original image. Feed to model and record probability. Compute AUC.
4. Aggregate mean and std AUC per method per domain.

#### Cross-Domain Consistency (`evaluate_cross_domain`)

For each method and each of the 81 classes:
1. Collect all cached heatmaps from the "real" domain for that class → average to produce a **real prototype** (mean heatmap).
2. Collect all cached heatmaps from the "sketch" domain → average to produce a **sketch prototype**.
3. Flatten both prototypes to 1D vectors and compute **cosine similarity**: `dot(real, sketch) / (||real|| × ||sketch||)`.
4. Save per-class, per-method cosine similarity to CSV.

#### Representation Analysis (`extract_features` + `plot_representations`)

1. Register a **forward hook** on `model.avgpool` to capture the 2048-D feature vector for each image.
2. Process all sampled images (both domains) in batches through the model.
3. Collect the 2048-D vectors along with their class labels and domain tags.
4. Run **UMAP** (`n_components=2, random_state=42`) and **t-SNE** (`perplexity=30, random_state=42`) on the feature matrix.
5. Generate 4 scatter plots: {UMAP, t-SNE} × {coloured by class, coloured by domain}.

---

## Results

### Classification Performance

| Trained on | Tested on | Accuracy | F1 Macro | F1 Weighted | Precision | Recall |
|------------|-----------|----------|----------|-------------|-----------|--------|
| Real | Real | **93.63%** | **0.9288** | **0.9365** | 0.9276 | **0.9334** |
| Real | Sketch | 56.17% | 0.5695 | 0.5837 | 0.6804 | 0.5755 |
| Sketch | Sketch | 80.62% | 0.8087 | 0.8057 | 0.8103 | 0.8178 |
| Sketch | Real | 83.02% | 0.8164 | 0.8313 | **0.8425** | 0.8193 |

#### Key observations

1. **Asymmetric transfer:** The sketch model transfers to real images (83.02%) far better than the real model transfers to sketches (56.17%) — a gap of ~27 percentage points.
2. **Shape bias:** Sketch training forces the model to learn *structural* features (edges, contours, spatial layout) that generalize across domains. Real training encourages reliance on *texture* features (colour gradients, surface patterns) that don't transfer to line drawings.
3. **The real→sketch drop** (37.46 pp) is consistent with the known texture bias of CNNs trained on photographs.
4. **Sketch→real actually gains** 2.4 pp over in-domain sketch performance, suggesting real images contain the structural features the sketch model learned, plus additional helpful information.

### Stability Results (SSIM)

| Method | Domain | Real Model (Mean ± Std) | Sketch Model (Mean ± Std) |
|--------|--------|------------------------|--------------------------|
| Grad-CAM | real | 0.942 ± 0.056 | 0.928 ± 0.078 |
| Grad-CAM | sketch | 0.896 ± 0.094 | 0.926 ± 0.079 |
| Grad-CAM++ | real | **0.946 ± 0.038** | **0.946 ± 0.042** |
| Grad-CAM++ | sketch | 0.927 ± 0.050 | 0.940 ± 0.043 |
| Int. Grad. | real | 0.563 ± 0.162 | 0.542 ± 0.181 |
| Int. Grad. | sketch | 0.555 ± 0.156 | 0.492 ± 0.170 |
| LIME | real | 0.471 ± 0.136 | 0.485 ± 0.139 |
| LIME | sketch | 0.471 ± 0.137 | 0.493 ± 0.143 |

#### Analysis

- **Grad-CAM++ is the most stable method** (SSIM ≥ 0.92 everywhere). Its higher-order gradient weighting provides natural robustness to perturbations.
- **Grad-CAM** is nearly as stable (~0.90–0.94), slightly lower on sketch images for the real model.
- **Integrated Gradients** has moderate stability (~0.49–0.56) with high variance. Its pixel-level sensitivity makes it responsive to small noise.
- **LIME** is the least stable (~0.47–0.49), reflecting its stochastic nature.
- The stability gap between CAM-based and pixel-level methods is expected: the 7×7 spatial resolution of CAM methods acts as a natural low-pass filter that smooths perturbation effects.
- The **sketch model shows more consistent stability** across domains (Grad-CAM real ≈ sketch), while the real model's stability drops on sketch images.

### Faithfulness Results

#### Deletion AUC (higher = more faithful)

| Method | Domain | Real Model (Mean ± Std) | Sketch Model (Mean ± Std) |
|--------|--------|------------------------|--------------------------|
| Grad-CAM | real | 0.294 ± 0.190 | 0.182 ± 0.164 |
| Grad-CAM | sketch | 0.107 ± 0.125 | 0.170 ± 0.159 |
| Grad-CAM++ | real | **0.323 ± 0.202** | **0.205 ± 0.178** |
| Grad-CAM++ | sketch | 0.125 ± 0.138 | 0.194 ± 0.172 |
| Int. Grad. | real | 0.277 ± 0.204 | 0.177 ± 0.197 |
| Int. Grad. | sketch | 0.125 ± 0.159 | 0.172 ± 0.193 |
| LIME | real | 0.152 ± 0.159 | 0.071 ± 0.098 |
| LIME | sketch | 0.054 ± 0.090 | 0.067 ± 0.093 |

#### Insertion AUC (higher = more faithful)

| Method | Domain | Real Model (Mean ± Std) | Sketch Model (Mean ± Std) |
|--------|--------|------------------------|--------------------------|
| Grad-CAM | real | 0.548 ± 0.134 | 0.473 ± 0.176 |
| Grad-CAM | sketch | 0.379 ± 0.174 | 0.492 ± 0.170 |
| Grad-CAM++ | real | 0.537 ± 0.144 | 0.453 ± 0.189 |
| Grad-CAM++ | sketch | 0.345 ± 0.189 | 0.470 ± 0.187 |
| Int. Grad. | real | 0.352 ± 0.188 | 0.321 ± 0.212 |
| Int. Grad. | sketch | 0.206 ± 0.177 | 0.272 ± 0.192 |
| LIME | real | **0.609 ± 0.105** | 0.550 ± 0.158 |
| LIME | sketch | 0.501 ± 0.165 | **0.571 ± 0.154** |

#### Analysis

- **LIME is the most faithful by insertion AUC** (0.61 real model / 0.57 sketch model in-domain). Its superpixel-based approach effectively isolates the regions the model depends on.
- **Grad-CAM++ leads on deletion AUC** (0.32 real model / 0.20 sketch model). Its attributions best pinpoint features whose removal disrupts predictions.
- **Integrated Gradients underperforms on faithfulness** despite pixel-level precision. Its distributed attributions across many pixels make it harder to isolate critical features.
- **Domain shift reduces faithfulness for the real model:** Deletion AUC drops from 0.29–0.32 (real) to 0.05–0.13 (sketch), indicating explanations become less meaningful on out-of-domain data.
- **The sketch model shows balanced faithfulness:** Unlike the real model, deletion and insertion scores are similar across domains (e.g., LIME insertion: 0.55 vs. 0.57), confirming shape-based representations produce more domain-invariant explanations.

### Cross-Domain Consistency Results (Cosine Similarity)

| Method | Real Model (Mean ± Std) | Sketch Model (Mean ± Std) |
|--------|------------------------|--------------------------|
| Grad-CAM | 0.953 ± 0.035 | 0.949 ± 0.039 |
| Grad-CAM++ | **0.963 ± 0.028** | **0.961 ± 0.033** |
| Int. Gradients | 0.842 ± 0.033 | 0.895 ± 0.024 |
| LIME | 0.871 ± 0.042 | 0.880 ± 0.041 |

#### Analysis

- **Grad-CAM++ achieves the highest consistency** (≥0.96 for both models). Its coarse resolution naturally produces similar heatmaps regardless of visual style.
- **Grad-CAM** is nearly as consistent (~0.95).
- **Integrated Gradients shows the largest model gap:** The sketch model (0.895) is substantially more consistent than the real model (0.842). This means the sketch model attends to the same structural features (edges, contours) in both domains, while the real model's pixel-level attributions are more domain-dependent.
- **LIME** achieves moderate consistency (~0.87–0.88), limited by stochastic superpixel segmentation.
- Overall high consistency (>0.84 for all methods) suggests that despite performance drops under domain shift, models still attend to broadly similar spatial regions.

### Representation Analysis (t-SNE / UMAP)

Feature vectors (2048-D) from the penultimate layer were projected to 2D using t-SNE and UMAP for both models.

#### Key findings

- **The real model separates domains:** In its feature space, real and sketch images form distinct clusters even for the same class. This domain separation explains the 37-point accuracy drop when transferring to sketches.
- **The sketch model mixes domains:** Real and sketch images are interleaved, with same-class images from both domains occupying overlapping regions. This domain-invariant organization explains the sketch model's superior transfer.
- **Class structure is preserved in both:** Both models produce coherent class clusters, confirming the 81-class classification task has been learned.
- These visualizations provide direct evidence for the **shape bias hypothesis**: sketch training creates a domain-agnostic feature space (shape-based), while real training encodes domain identity (texture-based) alongside class identity.

### Summary of Findings

| Criterion | Best Method | Key Insight |
|-----------|-------------|-------------|
| **Stability** | Grad-CAM++ (SSIM ≥ 0.92) | CAM-based methods are ~2× more stable than pixel-level methods due to spatial smoothing |
| **Faithfulness (Deletion)** | Grad-CAM++ (AUC 0.32) | Best at identifying features whose removal disrupts predictions |
| **Faithfulness (Insertion)** | LIME (AUC 0.61) | Superpixel regions most efficiently recover model confidence |
| **Cross-Domain Consistency** | Grad-CAM++ (cos. sim. 0.96) | Coarse resolution produces consistent attention across domains |
| **Domain-Invariant Representations** | Sketch model | Sketch training produces mixed real/sketch feature space; real training separates them |
| **Cross-Domain Transfer** | Sketch→Real (83.02%) | Shape-based features generalize; texture-based features don't |

**No single method wins on all axes.** Grad-CAM++ is the most stable and consistent but LIME is the most faithful. Integrated Gradients provides the finest spatial detail but at the cost of stability. This demonstrates why multi-axis evaluation is essential for reliable XAI assessment.

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

## Hyperparameters

| Category | Parameter | Value |
|----------|-----------|-------|
| **Data** | Number of classes | 81 |
| | Train/val/test split | 80/10/10 |
| | Image size | 224×224 |
| **Training** | Backbone | ResNet-152 (ImageNet-1K V2) |
| | Optimizer | AdamW |
| | Base learning rate | 10⁻³ |
| | Weight decay | 5×10⁻⁴ |
| | Batch size | 256 |
| | Max epochs | 100 |
| | Early stopping patience | 10 |
| | Warmup epochs | 5 |
| | Mixup α | 0.2 |
| | Label smoothing | 0.1 |
| | Freeze epochs | 5 |
| **XAI** | Samples per class/domain | 20 |
| | IG interpolation steps | 100 |
| | LIME perturbation samples | 3,000 |
| | LIME superpixels | ~150 |
| | Stability perturbations | 5 |

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
