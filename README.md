# Explainable AI on DomainNet: Cross-Domain Attribution Analysis

A deep learning + explainability research project that trains **ResNet-152** classifiers on the [DomainNet](http://ai.bu.edu/M3SDA/) benchmark and rigorously evaluates **four XAI attribution methods** across two visual domains — photographic (*real*) and hand-drawn (*sketch*).

The central question this project answers:

> *When a neural network is trained on one visual style, do its explanations remain consistent, faithful, and stable — even on images it has never seen before?*

<p align="center">
  <img src="docs/images/attribution_maps/gradcam_real_airplane.png" width="45%" alt="Grad-CAM on real airplane"/>
  <img src="docs/images/attribution_maps/lime_real_airplane.png" width="45%" alt="LIME on real airplane"/>
</p>
<p align="center"><em>Grad-CAM (left) and LIME (right) attribution maps on a real airplane — highlighting where the model looks to make its prediction.</em></p>

---

## Table of Contents

- [Why This Project?](#why-this-project)
- [Background & Related Work](#background--related-work)
- [Dataset & Classes](#dataset--classes)
- [Project Structure](#project-structure)
- [Pipeline at a Glance](#pipeline-at-a-glance)
- [Step 1 — Data Preparation (`prepare_data.py`)](#step-1--data-preparation-prepare_datapy)
- [Step 2 — Model Training (`train.py`)](#step-2--model-training-trainpy)
- [Step 3 — Evaluation (`evaluate.py`)](#step-3--evaluation-evaluatepy)
- [Step 4 — XAI Explainability (`explain.py`)](#step-4--xai-explainability-explainpy)
  - [Method 1: Grad-CAM](#method-1-grad-cam)
  - [Method 2: Grad-CAM++](#method-2-grad-cam-1)
  - [Method 3: Integrated Gradients](#method-3-integrated-gradients)
  - [Method 4: LIME](#method-4-lime)
  - [Evaluation Axis 1: Stability](#evaluation-axis-1-stability)
  - [Evaluation Axis 2: Faithfulness](#evaluation-axis-2-faithfulness)
  - [Evaluation Axis 3: Cross-Domain Consistency](#evaluation-axis-3-cross-domain-consistency)
  - [Evaluation Axis 4: Representation Behavior](#evaluation-axis-4-representation-behavior)
- [How the Training Went](#how-the-training-went)
- [Classification Results](#classification-results)
- [What the Models See: Attribution Map Examples](#what-the-models-see-attribution-map-examples)
- [Cross-Domain Attribution Comparisons](#cross-domain-attribution-comparisons)
- [Stability Results](#stability-results)
- [Faithfulness Results](#faithfulness-results)
- [Cross-Domain Consistency Results](#cross-domain-consistency-results)
- [Representation Analysis](#representation-analysis)
- [Discussion: What It All Means](#discussion-what-it-all-means)
- [Summary of Findings](#summary-of-findings)
- [Future Work](#future-work)
- [References](#references)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [Running on SLURM (`run_real.sh` / `run_sketch.sh`)](#running-on-slurm-run_realsh--run_sketchsh)
- [Hyperparameters](#hyperparameters)
- [Output Structure](#output-structure)
- [Installation](#installation)

---

## Why This Project?

Deep neural networks achieve state-of-the-art performance on visual recognition tasks, yet their internal decision processes remain opaque. Explainable AI (XAI) methods attempt to open this "black box" by attributing predictions to input features, but the reliability of these explanations is often taken for granted.

A critical challenge in deploying deep learning models is **domain shift** — the performance degradation that occurs when a model trained on one data distribution is applied to a different but related distribution. Domain shift is pervasive in real-world applications: a medical imaging model trained on data from one hospital may fail when applied to images from another; an autonomous driving system trained in sunny conditions may struggle in rain or fog.

Understanding *how* a model's reasoning changes under domain shift is arguably even more important than measuring the accuracy drop itself, because it reveals whether the model has learned genuinely transferable concepts or merely domain-specific shortcuts.

This project investigates these questions using the **DomainNet** benchmark, which contains images of the same object categories rendered in six distinct visual styles. By training separate classifiers on the *real* (photographic) and *sketch* (hand-drawn) domains and then applying multiple XAI techniques, we systematically study:

- **How domain shift affects model behaviour:** Does the model still focus on semantically meaningful object parts when shown an unfamiliar visual style, or does it rely on domain-specific artifacts?
- **How domain shift affects explanations:** Do attribution maps remain stable, faithful, and consistent when the input domain changes?
- **What features transfer across domains:** By comparing real-trained and sketch-trained models, we can identify whether texture-based or shape-based features lead to more robust reasoning under distribution shift.
- **Which XAI methods are most reliable:** Some explanation methods may appear informative in-domain but produce misleading or unstable attributions under domain shift, making them unreliable for safety-critical applications.

### What This Project Contributes

1. **Cross-domain classification pipeline:** Two independent ResNet-152 models trained with modern techniques (mixup, cosine annealing, differential learning rates, frozen warm-up) on balanced 81-class subsets of DomainNet, evaluated in both in-domain and cross-domain settings.
2. **Four-axis XAI evaluation under domain shift:** A rigorous framework that measures stability, faithfulness, cross-domain consistency, and representation behaviour for four attribution methods, specifically designed to test how explanations degrade when the input distribution changes.
3. **Comparative analysis:** Quantitative comparison of Grad-CAM, Grad-CAM++, Integrated Gradients, and LIME across two visual domains, revealing trade-offs between explanation granularity, stability, and faithfulness — and demonstrating that explanation reliability is *not* guaranteed even when classification accuracy remains high.
4. **Evidence for the shape bias hypothesis:** Our results provide direct evidence that sketch-trained models learn more transferable, shape-based representations, while photo-trained models exhibit texture bias that degrades both predictions and explanations under domain shift.

---

## Background & Related Work

**Grad-CAM** (Selvaraju et al., 2017) computes class-discriminative localisation maps using the gradients flowing into the final convolutional layer. **Grad-CAM++** (Chattopadhay et al., 2018) improves upon this with pixel-wise weighting using higher-order gradients, achieving better multi-instance localisation. **Integrated Gradients** (Sundararajan et al., 2017) provides axiomatic attributions by integrating gradients along a path from a baseline to the input. **LIME** (Ribeiro et al., 2016) is a model-agnostic method that fits a local linear surrogate model on perturbed versions of the input.

**DomainNet** (Peng et al., 2019) is a large-scale benchmark for multi-source domain adaptation containing ~0.6 million images across 345 categories and 6 visual domains. Prior work has used it primarily for domain adaptation and transfer learning; our focus on XAI evaluation across domains is novel.

Geirhos et al. (2019) demonstrated that ImageNet-trained CNNs exhibit a strong **texture bias**, relying on local texture patterns rather than global shape for classification. This finding motivates our comparison of real-trained (texture-rich) and sketch-trained (shape-only) models — and turned out to be one of the most important insights from this project.

---

## Dataset & Classes

**DomainNet** contains images of everyday objects across six visual domains: *clipart*, *infograph*, *painting*, *quickdraw*, *real*, and *sketch*. This project focuses on two domains that represent opposite ends of the visual abstraction spectrum:

| Domain   | Description                             | Visual Style         |
|----------|-----------------------------------------|----------------------|
| `real`   | Photographs from the internet           | Photo-realistic      |
| `sketch` | Hand-drawn pencil / line-art sketches   | Abstract / structural |

**81 classes** were selected from the original 345. Two independent models are trained — one per domain — and then cross-evaluated and explained. The XAI pipeline applies four attribution methods and measures each along four quantitative axes, producing a complete picture of *how* and *how reliably* the models reason.

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

After balancing, the final dataset splits look like this:

| Domain | Train  | Val   | Test  |
|--------|--------|-------|-------|
| Real   | 29,501 | 3,515 | 3,515 |
| Sketch | 29,501 | 2,724 | 2,724 |

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
   - Uses `sklearn.model_selection.train_test_split` with `stratify=labels` and a fixed seed (`random_state=42`)
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

Transfer learning from a pretrained backbone requires care: the new classification head is randomly initialised, so its initial gradients are essentially random noise. If these noisy gradients propagate back through the entire pretrained network from the start, they can corrupt the carefully learned low-level features (edges, textures, colour detectors) in the early layers.

To prevent this, the early layers are **frozen** during the first 5 epochs:

```
Frozen:    conv1, bn1, layer1, layer2  (first 5 epochs)
Trainable: layer3, layer4, fc         (all epochs)
```

This allows the new FC head and the higher-level feature layers (`layer3`, `layer4`) to adapt to the 81-class task without destroying the universal low-level features. After epoch 5, when the head has stabilised, all layers are unfrozen and the full network fine-tunes end-to-end. This two-phase approach consistently outperforms training all layers from the start, especially when the target dataset is substantially different from ImageNet.

#### Differential learning rates

Even after unfreezing, the pretrained backbone and the new head should not be updated at the same rate. The backbone already contains useful features learned from 1.2 million ImageNet images; aggressive updates would cause **catastrophic forgetting** — the phenomenon where a neural network forgets previously learned knowledge when trained on new data.

- Backbone parameters: `lr × 0.1`
- FC head parameters: `lr` (full learning rate)

This ensures the backbone parameters are gently nudged toward the new task rather than overwritten, while the FC head can learn quickly from scratch.

#### Cosine annealing with linear warmup

The learning rate schedule combines two phases:

```
Epochs 0–4:  Linear warmup   (lr ramps from 0 → lr)
Epochs 5–N:  Cosine decay    (lr decays from lr → 0.01 × lr)
```

During **linear warmup**, the learning rate ramps from near-zero to the full value. This prevents the optimiser from making large, poorly-directed updates before it has accumulated meaningful gradient statistics. After warmup, **cosine annealing** smoothly decays the learning rate following a half-cosine curve. Cosine annealing is preferred over step decay because it avoids sudden learning rate drops that can destabilise training, and its smooth decay allows the model to fine-tune with increasingly small updates as it converges.

#### Mixup augmentation

At each training step, two random samples are blended:

```python
mixed_x = λ × x_i + (1 − λ) × x_j    where λ ~ Beta(0.2, 0.2)
loss = λ × CE(pred, y_i) + (1 − λ) × CE(pred, y_j)
```

The Beta(0.2, 0.2) distribution is U-shaped, meaning λ tends to be close to 0 or 1 — so most mixed images look mostly like one of the two originals, with a subtle blend of the other. Mixup serves several purposes:

- **Regularisation:** Prevents the model from memorising individual training images, reducing overfitting.
- **Smoother decision boundaries:** The model learns to make gradual transitions between classes rather than sharp, brittle boundaries.
- **Improved cross-domain generalisation:** By training on interpolated images, the model is exposed to visual variations not present in the original dataset.

Note that mixup causes training accuracy to appear lower than validation accuracy — this is expected, since the model is trained on blended images but evaluated on clean ones.

#### Weighted CrossEntropy + label smoothing

- **Cross-entropy loss** measures the divergence between the predicted probability distribution and the true label. For a single sample with true class c: `CE = -log(p_c)`.
- **Class weighting** addresses class imbalance by multiplying each sample's loss by its class weight. Classes with fewer training samples receive higher weights, so the model pays more attention to rare classes.
- **Label smoothing** (ε = 0.1) replaces the hard one-hot target with a softened distribution, preventing the model from becoming overconfident. Instead of driving logits to extreme values (which can cause gradient saturation and poor calibration), it encourages the model to maintain a small amount of uncertainty. Label smoothing acts as a regulariser and typically improves generalisation performance, especially under domain shift.

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

Validation and test images use: `Resize(256) → CenterCrop(224) → ToTensor → Normalize`.

#### Early stopping & checkpointing

- Best model saved by **macro F1** on the validation set (rather than accuracy, because macro F1 equally weighs all classes regardless of support).
- Checkpoint saved every epoch to allow training resumption after interruption.
- Early stopping triggers after `patience=10` epochs with no validation loss improvement, preventing wasted computation.

### Computational Resources

Training was performed on an HPC cluster using SLURM:

| Resource  | Allocation           |
|-----------|----------------------|
| Partition | H200                 |
| Nodes     | 1                    |
| CPUs      | 16                   |
| GPUs      | 8 × NVIDIA H200      |
| Memory    | 200 GB               |

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

In this project we apply four complementary attribution methods. Each takes a different approach to the same question: *"Which pixels drove the model to predict this class?"* Understanding the differences between these methods is essential for interpreting the results.

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

<p align="center">
  <img src="docs/images/attribution_maps/gradcam_real_airplane.png" width="80%" alt="Grad-CAM example"/>
</p>
<p align="center"><em>Grad-CAM on a real airplane — the broad red region correctly covers the aircraft body, but fine details are lost at 7×7 resolution.</em></p>

---

### Method 2: Grad-CAM++

An improved version of Grad-CAM that addresses two practical problems: (1) Grad-CAM often highlights only the most discriminative part of an object (face of a dog but not the body), and (2) it struggles when multiple instances of the same class appear (a flock of birds).

#### The core improvement

The only change from Grad-CAM is *how the channel importance weights are computed*. Instead of simple global average pooling of gradients, Grad-CAM++ uses **higher-order derivatives** (second and third derivatives of the class score) to compute pixel-wise weights.

#### Intuition

Grad-CAM asks: "On average across all positions, does this channel help predict dog?" Grad-CAM++ asks: "At *each specific position*, how much does this pixel contribute to the positive evidence for dog?" By considering second-order derivatives (curvature), the method captures not just *whether* the gradient is positive but *how quickly it is changing*. The practical effect is that Grad-CAM++ distributes attention more evenly across the entire object:

- Grad-CAM on a dog → highlights only the face
- Grad-CAM++ on the same dog → highlights face, body, legs, and tail

**Strengths:** Better full-object coverage, more numerically stable weights, better multi-instance localization, same computational cost as Grad-CAM (one forward + backward pass).

**Limitations:** Still limited to 7×7 resolution, higher-order gradients can be unstable for some architectures (works well for ResNets), still a heuristic without formal guarantees.

<p align="center">
  <img src="docs/images/attribution_maps/gradcampp_real_airplane.png" width="80%" alt="Grad-CAM++ example"/>
</p>
<p align="center"><em>Grad-CAM++ on the same airplane — note the broader, more complete coverage of all aircraft compared to Grad-CAM above.</em></p>

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

<p align="center">
  <img src="docs/images/attribution_maps/ig_real_airplane.png" width="80%" alt="Integrated Gradients example"/>
</p>
<p align="center"><em>Integrated Gradients on the same airplane — pixel-level detail picks up individual edges of the fuselage and wings, but the output is sparser and noisier.</em></p>

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

<p align="center">
  <img src="docs/images/attribution_maps/lime_real_airplane.png" width="80%" alt="LIME example"/>
</p>
<p align="center"><em>LIME on the same airplane — superpixel regions are coloured by importance. The airplane bodies are clearly identified as the most important regions (warm colours).</em></p>

---

### All Four Methods Side-by-Side

<p align="center">
  <img src="docs/images/attribution_maps/gradcam_real_skull.png" width="45%" alt="Grad-CAM skull"/>
  <img src="docs/images/attribution_maps/gradcampp_real_skull.png" width="45%" alt="Grad-CAM++ skull"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/ig_real_skull.png" width="45%" alt="IG skull"/>
  <img src="docs/images/attribution_maps/lime_real_skull.png" width="45%" alt="LIME skull"/>
</p>
<p align="center"><em>All four methods on a real skull (100% accuracy class). Top: Grad-CAM (left), Grad-CAM++ (right). Bottom: Integrated Gradients (left), LIME (right). All methods agree on the key features — eye sockets, nasal cavity, cranium outline.</em></p>

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

**In summary:** Grad-CAM and Grad-CAM++ are fast, stable, and produce coarse heatmaps that answer "*where* is the model looking?" Integrated Gradients is slower but provides pixel-level detail and mathematical guarantees. LIME is the most flexible (model-agnostic) but the slowest and least stable. By using all four, we obtain a comprehensive view of the model's reasoning from multiple perspectives.

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

## How the Training Went

### Real-Domain Model

The real-domain model was trained for **21 epochs** before early stopping kicked in. It converged quickly, reaching a peak validation F1 (macro) of **0.9451** at epoch 9. The validation accuracy stabilised around 94.5% from epoch 5 onward, which is when the frozen layers were unfrozen.

One interesting observation: because of mixup augmentation, the training accuracy was consistently *lower* than the validation accuracy. This is completely expected — the model trains on blended images but is evaluated on clean ones — but it can look confusing at first glance. For example, at epoch 21 the training accuracy was 88.88% while validation accuracy was 94.22%.

The validation loss plateaued around epoch 11 (~1.047), and the model continued for 10 more epochs (the patience window) before early stopping triggered at epoch 21.

<p align="center">
  <img src="docs/images/evaluation/real_training_summary.png" width="80%" alt="Real model training summary"/>
</p>
<p align="center"><em>Real model training summary — loss, accuracy, F1, and precision/recall over 21 epochs. Validation accuracy stabilizes around 94.5% from epoch 5.</em></p>

### Sketch-Domain Model

The sketch-domain model was trained for **19 epochs**, achieving a peak validation F1 (macro) of **0.8336** at epoch 13. The convergence was noticeably slower and the final performance was lower than the real model — reflecting the greater challenge of learning from abstract sketches. Where real photographs provide rich texture, colour, and contextual cues, sketches offer only line strokes and shape outlines.

The validation accuracy plateaued around 82–83%, with training accuracy at 87.29% by epoch 18. The gap between training and validation is smaller here than for the real model, suggesting the sketch domain leaves less room for the model to overfit.

<p align="center">
  <img src="docs/images/evaluation/sketch_training_summary.png" width="80%" alt="Sketch model training summary"/>
</p>
<p align="center"><em>Sketch model training summary — 19 epochs. Slower convergence and lower final accuracy (82-83%) reflects the greater challenge of learning from abstract sketches.</em></p>

---

## Classification Results

### Overall Performance

| Trained on | Tested on | Accuracy | F1 Macro | F1 Weighted | Precision | Recall |
|------------|-----------|----------|----------|-------------|-----------|--------|
| Real | Real | **93.63%** | **0.9288** | **0.9365** | 0.9276 | **0.9334** |
| Real | Sketch | 56.17% | 0.5695 | 0.5837 | 0.6804 | 0.5755 |
| Sketch | Sketch | 80.62% | 0.8087 | 0.8057 | 0.8103 | 0.8178 |
| Sketch | Real | 83.02% | 0.8164 | 0.8313 | **0.8425** | 0.8193 |

### Confusion Matrices

The confusion matrices tell a vivid story of how the two models handle in-domain vs. cross-domain data.

<p align="center">
  <img src="docs/images/evaluation/cm_real_real.png" width="45%" alt="Real→Real CM"/>
  <img src="docs/images/evaluation/cm_real_sketch.png" width="45%" alt="Real→Sketch CM"/>
</p>
<p align="center"><em>Real model — In-domain (left, strong diagonal) vs. Cross-domain on sketches (right, significant off-diagonal confusion). The real model excels on photographs but falls apart on sketches.</em></p>

<p align="center">
  <img src="docs/images/evaluation/cm_sketch_sketch.png" width="45%" alt="Sketch→Sketch CM"/>
  <img src="docs/images/evaluation/cm_sketch_real.png" width="45%" alt="Sketch→Real CM"/>
</p>
<p align="center"><em>Sketch model — In-domain on sketches (left) vs. Cross-domain on real (right). The sketch→real confusion matrix remains surprisingly clean, showing the sketch model transfers well.</em></p>

### Top and Bottom Performing Classes

#### Real Model (In-Domain: Real → Real)

| **Top 10** | **Acc%** | **F1** | | **Bottom 10** | **Acc%** | **F1** |
|---|---|---|---|---|---|---|
| The Mona Lisa | 100.0 | 0.983 | | saw | 66.67 | 0.696 |
| basketball | 100.0 | 1.000 | | eye | 75.36 | 0.846 |
| saxophone | 100.0 | 1.000 | | circle | 80.77 | 0.857 |
| skull | 100.0 | 1.000 | | spoon | 81.13 | 0.878 |
| trumpet | 100.0 | 1.000 | | triangle | 81.58 | 0.838 |
| backpack | 100.0 | 0.936 | | flower | 83.33 | 0.896 |
| banana | 100.0 | 0.963 | | barn | 83.87 | 0.881 |
| baseball | 100.0 | 0.900 | | shovel | 84.44 | 0.905 |
| firetruck | 100.0 | 0.974 | | feather | 86.00 | 0.860 |
| spreadsheet | 100.0 | 0.987 | | suitcase | 86.05 | 0.914 |

The real model excels at classes with highly distinctive visual features: *saxophone* has a unique metallic curved shape, *skull* has unmistakable eye sockets and jaw structure, and *trumpet* has a distinctive bell shape. The bottom performers reveal interesting failure patterns: *saw* (66.67%) is often confused with other elongated tools; *circle* and *triangle* lack the rich texture cues that the real model relies on; and *spoon* shares visual similarity with other utensils.

#### Sketch Model (In-Domain: Sketch → Sketch)

| **Top 10** | **Acc%** | **F1** | | **Bottom 10** | **Acc%** | **F1** |
|---|---|---|---|---|---|---|
| banana | 100.0 | 0.952 | | shovel | 44.44 | 0.544 |
| barn | 100.0 | 0.930 | | scorpion | 55.56 | 0.694 |
| baseball | 100.0 | 0.833 | | sword | 55.26 | 0.568 |
| bicycle | 100.0 | 1.000 | | firetruck | 57.58 | 0.623 |
| marker | 100.0 | 0.828 | | triangle | 60.00 | 0.667 |
| axe | 95.45 | 0.840 | | saw | 63.64 | 0.667 |
| birthday cake | 95.65 | 0.936 | | palm tree | 64.71 | 0.550 |
| light bulb | 95.00 | 0.905 | | sock | 64.44 | 0.699 |
| tiger | 94.87 | 0.925 | | circle | 65.00 | 0.578 |
| laptop | 93.75 | 0.896 | | scissors | 65.91 | 0.674 |

The sketch model's top performers include classes with structurally distinctive shapes that are easy to draw consistently: *bicycle* has its iconic two-wheel-and-frame structure, *banana* has a unique curved shape, and *barn* has a recognisable rooftop silhouette. The bottom performers reveal that sketch classification is hardest for **elongated tools** (shovel at 44.44%, sword at 55.26%, saw at 63.64%) whose sketches look similar, and **simple geometric shapes** (triangle at 60.00%, circle at 65.00%) that lack distinguishing internal detail.

Interestingly, *firetruck* drops from 100% in the real model to 57.58% in the sketch model — likely because sketches of firetrucks lack the distinctive red colour and detailed ladder/hose features that make them easy to identify in photographs. This is a perfect example of texture bias in action.

### Key Observations

1. **Asymmetric transfer:** The sketch model transfers to real images (83.02%) far better than the real model transfers to sketches (56.17%) — a gap of ~27 percentage points. This was the most striking finding of the project.
2. **Shape bias:** Sketch training forces the model to learn *structural* features (edges, contours, spatial layout) that generalize across domains. Real training encourages reliance on *texture* features (colour gradients, surface patterns) that don't transfer to line drawings.
3. **The real→sketch drop** (37.46 pp) is consistent with the known texture bias of CNNs trained on photographs (Geirhos et al., 2019).
4. **Sketch→real actually gains** 2.4 pp over in-domain sketch performance, suggesting real images contain the structural features the sketch model learned, plus additional helpful information.

---

## What the Models See: Attribution Map Examples

For each model, attribution maps were generated for 20 images per class per domain (1,560 images per domain, 3,120 total per model). Each attribution is saved as a three-panel image: original | heatmap (jet colourmap) | overlay. Below are representative examples that illustrate how the four methods compare and what the models focus on.

### Real Airplane — All Four Methods (Real Model)

<p align="center">
  <img src="docs/images/attribution_maps/gradcam_real_airplane.png" width="45%" alt="Grad-CAM airplane"/>
  <img src="docs/images/attribution_maps/gradcampp_real_airplane.png" width="45%" alt="Grad-CAM++ airplane"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/ig_real_airplane.png" width="45%" alt="IG airplane"/>
  <img src="docs/images/attribution_maps/lime_real_airplane.png" width="45%" alt="LIME airplane"/>
</p>
<p align="center"><em>All four methods on a real airplane (real model). Top: Grad-CAM (left), Grad-CAM++ (right). Bottom: Integrated Gradients (left), LIME (right). Grad-CAM and Grad-CAM++ produce broad, focused heatmaps centred on the aircraft body. Integrated Gradients provides pixel-level detail, highlighting specific edges of the fuselage and wings. LIME identifies superpixel regions covering the body and sky boundary. All four methods agree on the general region of interest.</em></p>

### Sketch Airplane — All Four Methods (Sketch Model)

<p align="center">
  <img src="docs/images/attribution_maps/sketch_model/gradcam_sketch_airplane.png" width="45%" alt="Grad-CAM sketch airplane"/>
  <img src="docs/images/attribution_maps/sketch_model/gradcampp_sketch_airplane.png" width="45%" alt="Grad-CAM++ sketch airplane"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/sketch_model/ig_sketch_airplane.png" width="45%" alt="IG sketch airplane"/>
  <img src="docs/images/attribution_maps/sketch_model/lime_sketch_airplane.png" width="45%" alt="LIME sketch airplane"/>
</p>
<p align="center"><em>All four methods on a sketch airplane (sketch model). On sketches, Grad-CAM focuses on the overall shape contour rather than texture details — consistent with the sketch model learning structural features. Integrated Gradients traces specific line strokes of the drawing. LIME highlights the superpixel regions that contain the drawn lines.</em></p>

### Real Tiger — All Four Methods (Real Model, 98.36% accuracy)

<p align="center">
  <img src="docs/images/attribution_maps/real_model/gradcam_real_tiger.png" width="45%" alt="Grad-CAM tiger"/>
  <img src="docs/images/attribution_maps/real_model/gradcampp_real_tiger.png" width="45%" alt="Grad-CAM++ tiger"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/real_model/ig_real_tiger.png" width="45%" alt="IG tiger"/>
  <img src="docs/images/attribution_maps/real_model/lime_real_tiger.png" width="45%" alt="LIME tiger"/>
</p>
<p align="center"><em>All four methods on a real tiger (real model). Grad-CAM and Grad-CAM++ focus on the face and striped body — the model correctly uses the distinctive stripe pattern and facial features. Integrated Gradients reveals that individual stripes and the eye region are the most important pixels. This confirms the real model leverages texture features (stripes) which are domain-specific and may not transfer to sketches.</em></p>

### Real Scissors — All Four Methods (Real Model, 98.25% accuracy)

<p align="center">
  <img src="docs/images/attribution_maps/real_model/gradcam_real_scissors.png" width="45%" alt="Grad-CAM scissors"/>
  <img src="docs/images/attribution_maps/real_model/gradcampp_real_scissors.png" width="45%" alt="Grad-CAM++ scissors"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/real_model/ig_real_scissors.png" width="45%" alt="IG scissors"/>
  <img src="docs/images/attribution_maps/real_model/lime_real_scissors.png" width="45%" alt="LIME scissors"/>
</p>
<p align="center"><em>All four methods on real scissors (real model). The scissor blades and handle junction are highlighted by all methods, confirming the model uses the correct structural features.</em></p>

### Real Lighthouse — All Four Methods (Real Model, 95.83% accuracy)

<p align="center">
  <img src="docs/images/attribution_maps/real_model/gradcam_real_lighthouse.png" width="45%" alt="Grad-CAM lighthouse"/>
  <img src="docs/images/attribution_maps/real_model/gradcampp_real_lighthouse.png" width="45%" alt="Grad-CAM++ lighthouse"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/real_model/ig_real_lighthouse.png" width="45%" alt="IG lighthouse"/>
  <img src="docs/images/attribution_maps/real_model/lime_real_lighthouse.png" width="45%" alt="LIME lighthouse"/>
</p>
<p align="center"><em>All four methods on a real lighthouse (real model). The tower structure and lantern room at the top are correctly identified as the key features.</em></p>

### Sketch Saxophone — All Four Methods (Sketch Model)

<p align="center">
  <img src="docs/images/attribution_maps/sketch_model/gradcam_sketch_saxophone.png" width="45%" alt="Grad-CAM saxophone"/>
  <img src="docs/images/attribution_maps/sketch_model/gradcampp_sketch_saxophone.png" width="45%" alt="Grad-CAM++ saxophone"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/sketch_model/ig_sketch_saxophone.png" width="45%" alt="IG saxophone"/>
  <img src="docs/images/attribution_maps/sketch_model/lime_sketch_saxophone.png" width="45%" alt="LIME saxophone"/>
</p>
<p align="center"><em>All four methods on a sketch saxophone (sketch model). The distinctive curved body and bell of the saxophone are highlighted. Integrated Gradients traces the specific ink strokes of the drawing with high precision.</em></p>

### Real Lion — Grad-CAM vs. Integrated Gradients (Real Model, 98.08% accuracy)

<p align="center">
  <img src="docs/images/attribution_maps/real_model/gradcam_real_lion.png" width="45%" alt="Grad-CAM lion"/>
  <img src="docs/images/attribution_maps/real_model/ig_real_lion.png" width="45%" alt="IG lion"/>
</p>
<p align="center"><em>Grad-CAM (left) highlights the face and mane region as a broad hot zone. Integrated Gradients (right) provides finer detail, picking up specific facial features like the eyes, nose, and mane edges. The model is clearly attending to the correct semantically meaningful features.</em></p>

---

## Cross-Domain Attribution Comparisons

One of the most important questions in this project is whether the real model and sketch model focus on the *same* features when looking at the same object class. The comparisons below reveal that the answer depends heavily on which model was trained on which domain.

### Sketch Bicycle: Real Model vs. Sketch Model

<p align="center">
  <img src="docs/images/attribution_maps/gradcam_real_model_sketch_bicycle.png" width="45%" alt="Real model on sketch bicycle"/>
  <img src="docs/images/attribution_maps/gradcam_sketch_model_sketch_bicycle.png" width="45%" alt="Sketch model on sketch bicycle"/>
</p>
<p align="center"><em>Grad-CAM on a sketch bicycle — Real model (left) vs. Sketch model (right). The sketch model produces a more focused, confident heatmap centred on the bicycle's structural features — wheels, frame, handlebars. The real model produces a more diffuse, uncertain heatmap, reflecting its difficulty in recognising the object without texture cues.</em></p>

### Sketch Skull: Real Model vs. Sketch Model

<p align="center">
  <img src="docs/images/attribution_maps/cross_domain/gradcam_real_model_sketch_skull.png" width="45%" alt="Real model Grad-CAM on sketch skull"/>
  <img src="docs/images/attribution_maps/cross_domain/gradcam_sketch_model_sketch_skull.png" width="45%" alt="Sketch model Grad-CAM on sketch skull"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/cross_domain/ig_real_model_sketch_skull.png" width="45%" alt="Real model IG on sketch skull"/>
  <img src="docs/images/attribution_maps/cross_domain/ig_sketch_model_sketch_skull.png" width="45%" alt="Sketch model IG on sketch skull"/>
</p>
<p align="center"><em>Sketch skull: real model (left) vs. sketch model (right). Top: Grad-CAM. Bottom: Integrated Gradients. The sketch model produces sharper, more confident attributions because it was trained on similar line-art images. The real model's attributions are more scattered, suggesting uncertainty about which features to rely on in an unfamiliar visual style.</em></p>

### Sketch Tiger: Real Model vs. Sketch Model

<p align="center">
  <img src="docs/images/attribution_maps/cross_domain/gradcam_real_model_sketch_tiger.png" width="45%" alt="Real model Grad-CAM on sketch tiger"/>
  <img src="docs/images/attribution_maps/cross_domain/gradcam_sketch_model_sketch_tiger.png" width="45%" alt="Sketch model Grad-CAM on sketch tiger"/>
</p>
<p align="center">
  <img src="docs/images/attribution_maps/cross_domain/lime_real_model_sketch_tiger.png" width="45%" alt="Real model LIME on sketch tiger"/>
  <img src="docs/images/attribution_maps/cross_domain/lime_sketch_model_sketch_tiger.png" width="45%" alt="Sketch model LIME on sketch tiger"/>
</p>
<p align="center"><em>Sketch tiger: real model (left) vs. sketch model (right). Top: Grad-CAM. Bottom: LIME. The real model struggles with the sketch — its heatmap is diffuse and may focus on background regions, indicating poor explanation quality under domain shift. The sketch model produces more targeted attributions on the tiger's body and stripes. This illustrates how domain shift degrades not just accuracy but also explanation quality.</em></p>

---

## Stability Results

**Question:** Do attribution maps stay consistent when the input is slightly perturbed?

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

### What the stability results tell us

- **Grad-CAM++ is the most stable method** (SSIM ≥ 0.92 everywhere). Its higher-order gradient weighting provides natural robustness to perturbations.
- **Grad-CAM** is nearly as stable (~0.90–0.94), slightly lower on sketch images for the real model.
- **Integrated Gradients** has moderate stability (~0.49–0.56) with high variance. Its pixel-level sensitivity makes it responsive to small noise.
- **LIME** is the least stable (~0.47–0.49), reflecting its stochastic nature and sensitivity to superpixel boundaries.
- The stability gap between CAM-based and pixel-level methods is expected: the 7×7 spatial resolution of CAM methods acts as a natural **low-pass filter** that smooths perturbation effects. It's the same reason why a blurry photo looks the same whether you add noise or not — the blurriness masks the noise.
- The **sketch model shows more consistent stability** across domains (Grad-CAM real ≈ sketch), while the real model's stability drops on sketch images. This is another sign of the real model's domain sensitivity.

---

## Faithfulness Results

**Question:** Do the attribution maps actually identify the pixels that the model relies on?

### Deletion AUC (higher = more faithful)

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

### Insertion AUC (higher = more faithful)

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

### What the faithfulness results tell us

- **LIME is the most faithful by insertion AUC** (0.61 real model / 0.57 sketch model in-domain). Its superpixel-based approach effectively isolates the regions the model depends on — when you reveal just those regions, the model's confidence recovers quickly.
- **Grad-CAM++ leads on deletion AUC** (0.32 real model / 0.20 sketch model). Its attributions best pinpoint features whose removal disrupts predictions.
- **Integrated Gradients underperforms on faithfulness** despite pixel-level precision. Its distributed attributions across many pixels make it harder to isolate critical features through sequential deletion/insertion.
- **Domain shift reduces faithfulness for the real model:** Deletion AUC drops from 0.29–0.32 (real) to 0.05–0.13 (sketch), indicating explanations become less meaningful on out-of-domain data. When the model doesn't know what to look for, its explanations become unreliable.
- **The sketch model shows balanced faithfulness:** Unlike the real model, deletion and insertion scores are similar across domains (e.g., LIME insertion: 0.55 vs. 0.57), confirming shape-based representations produce more domain-invariant explanations.

---

## Cross-Domain Consistency Results

**Question:** When the same class is viewed in different visual styles, do the attribution maps highlight similar spatial regions?

| Method | Real Model (Mean ± Std) | Sketch Model (Mean ± Std) |
|--------|------------------------|--------------------------|
| Grad-CAM | 0.953 ± 0.035 | 0.949 ± 0.039 |
| Grad-CAM++ | **0.963 ± 0.028** | **0.961 ± 0.033** |
| Int. Gradients | 0.842 ± 0.033 | 0.895 ± 0.024 |
| LIME | 0.871 ± 0.042 | 0.880 ± 0.041 |

### What the consistency results tell us

- **Grad-CAM++ achieves the highest consistency** (≥0.96 for both models). Its coarse resolution naturally produces similar heatmaps regardless of visual style — it's hard for a 7×7 grid to look very different between a photo and a sketch of the same object.
- **Grad-CAM** is nearly as consistent (~0.95).
- **Integrated Gradients shows the largest model gap:** The sketch model (0.895) is substantially more consistent than the real model (0.842). This means the sketch model attends to the same structural features (edges, contours) in both domains, while the real model's pixel-level attributions are more domain-dependent — it looks for texture in photos and doesn't know what to look for in sketches.
- **LIME** achieves moderate consistency (~0.87–0.88), limited by stochastic superpixel segmentation.
- Overall high consistency (>0.84 for all methods) suggests that despite performance drops under domain shift, models still attend to broadly similar spatial regions. The models aren't completely lost — they still roughly know where to look, but their confidence and precision suffer.

---

## Representation Analysis

**Question:** How does the model organize images in its internal feature space?

Feature vectors (2048-D) from the penultimate layer were projected to 2D using t-SNE and UMAP for both models. The results are revealing.

### Real Model — Feature Space

<p align="center">
  <img src="docs/images/representations/real_tsne_domain.png" width="45%" alt="Real model t-SNE by domain"/>
  <img src="docs/images/representations/real_umap_domain.png" width="45%" alt="Real model UMAP by domain"/>
</p>
<p align="center"><em>Real model feature space coloured by domain — t-SNE (left) and UMAP (right). Blue = real, orange = sketch. Notice the clear separation between domains — the real model encodes domain-specific (texture-based) features.</em></p>

### Sketch Model — Feature Space

<p align="center">
  <img src="docs/images/representations/sketch_tsne_domain.png" width="45%" alt="Sketch model t-SNE by domain"/>
  <img src="docs/images/representations/sketch_umap_domain.png" width="45%" alt="Sketch model UMAP by domain"/>
</p>
<p align="center"><em>Sketch model feature space coloured by domain — t-SNE (left) and UMAP (right). The real and sketch points are much more interleaved, confirming the sketch model learns domain-invariant (shape-based) representations.</em></p>

### Class Structure

<p align="center">
  <img src="docs/images/representations/real_tsne_class.png" width="45%" alt="Real model t-SNE by class"/>
  <img src="docs/images/representations/sketch_tsne_class.png" width="45%" alt="Sketch model t-SNE by class"/>
</p>
<p align="center"><em>t-SNE coloured by class — Real model (left) vs. Sketch model (right). Both models produce coherent class clusters, confirming the 81-class task has been learned successfully.</em></p>

### What the representations reveal

This is where the story comes together. The representation analysis provides the *mechanistic explanation* for why the two models transfer so differently:

- **The real model separates domains:** In its feature space, real and sketch images form distinct clusters even for the same class. The model has learned separate representations for "photo of airplane" and "sketch of airplane". This domain separation directly explains the 37-point accuracy drop when transferring to sketches — the model maps sketch images to unfamiliar locations in feature space.
- **The sketch model mixes domains:** Real and sketch images are interleaved, with same-class images from both domains occupying overlapping regions. The model has learned a single, unified representation for "airplane" regardless of whether it's a photo or a sketch. This domain-invariant organization explains the sketch model's superior transfer.
- **Class structure is preserved in both:** Despite differences in domain separation, both models produce coherent class clusters, confirming that the 81-class classification task has been learned.
- These visualizations provide **direct evidence for the shape bias hypothesis**: sketch training creates a domain-agnostic feature space (shape-based), while real training encodes domain identity (texture-based) alongside class identity.

---

## Discussion: What It All Means

### Domain Transfer Asymmetry

The most striking finding is the **asymmetric transfer** between domains. The sketch→real transfer (83.02%) substantially outperforms real→sketch (56.17%). This result aligns with the *texture bias* hypothesis: CNNs trained on photographs tend to rely on texture cues (colour gradients, surface patterns) that are absent in line drawings. Conversely, sketch training forces the model to rely on *shape-based* features (edges, contours, spatial relationships) that are present in both domains.

The real model loses 37.46 percentage points when transferred to sketches. The sketch model actually *gains* 2.40 percentage points when transferred to real images. This asymmetry is remarkable — it means learning from abstract line drawings produces a more universally useful representation than learning from rich, detailed photographs.

### Stability vs. Granularity Trade-off

There is a clear trade-off between explanation stability and spatial granularity:

- **CAM-based methods** (Grad-CAM, Grad-CAM++) operate at 7×7 resolution, which acts as a natural low-pass filter. This makes them highly stable (SSIM > 0.90) but unable to capture fine-grained details.
- **Integrated Gradients** operates at full 224×224 resolution, capturing pixel-level details but at the cost of perturbation sensitivity (SSIM ≈ 0.50).
- **LIME** operates at superpixel resolution, with inherent stochasticity from its sampling-based approach.

This trade-off is fundamental — you can't have both high resolution and high stability. Practitioners need to choose based on their application: if you need robust, consistent explanations (e.g., for regulatory compliance), use Grad-CAM++. If you need to understand exactly which pixels matter (e.g., for debugging), use Integrated Gradients and accept the noise.

### Explanation Quality Under Domain Shift

Our cross-domain attribution comparisons and the quantitative faithfulness results confirm that **domain shift degrades explanation quality**, not just classification accuracy. When the real model is applied to sketches:

- Attribution maps become more diffuse and less focused on the object.
- Faithfulness drops sharply: deletion AUC falls from 0.29–0.32 to 0.05–0.13, indicating the explanations no longer identify the model's actual decision features.
- LIME and Integrated Gradients show more scattered attributions, suggesting the model is "guessing" rather than using learned features.

In contrast, the sketch model's attributions on real images remain relatively focused, and its faithfulness scores are balanced across domains (e.g., LIME insertion: 0.55 real vs. 0.57 sketch). The representation analysis provides a mechanistic explanation: the sketch model's feature space mixes domains, while the real model separates them, directly explaining the asymmetric transfer and explanation degradation.

### Recommendations for Practitioners

Based on our analysis:

- **For quick, reliable explanations:** Use Grad-CAM++ — it offers the best stability while providing class-discriminative spatial highlighting.
- **For detailed pixel-level analysis:** Use Integrated Gradients — despite lower stability, its axiomatic guarantees make it suitable for rigorous analysis.
- **For model-agnostic explanations:** Use LIME — when the model architecture is unknown, LIME is the only option, though users should be aware of its stochastic nature.
- **For cross-domain studies:** Consider training on sketches or shape-biased data to improve both classification transfer and explanation consistency.

---

## Summary of Findings

| Criterion | Best Method | Key Insight |
|-----------|-------------|-------------|
| **Stability** | Grad-CAM++ (SSIM ≥ 0.92) | CAM-based methods are ~2× more stable than pixel-level methods due to spatial smoothing |
| **Faithfulness (Deletion)** | Grad-CAM++ (AUC 0.32) | Best at identifying features whose removal disrupts predictions |
| **Faithfulness (Insertion)** | LIME (AUC 0.61) | Superpixel regions most efficiently recover model confidence |
| **Cross-Domain Consistency** | Grad-CAM++ (cos. sim. 0.96) | Coarse resolution produces consistent attention across domains |
| **Domain-Invariant Representations** | Sketch model | Sketch training produces mixed real/sketch feature space; real training separates them |
| **Cross-Domain Transfer** | Sketch→Real (83.02%) | Shape-based features generalize; texture-based features don't |

**No single method wins on all axes.** Grad-CAM++ is the most stable and consistent but LIME is the most faithful. Integrated Gradients provides the finest spatial detail but at the cost of stability. This demonstrates why multi-axis evaluation is essential — any single metric would give a misleading picture of explanation quality.

The overarching takeaway: **XAI evaluation must be multi-axis.** A comprehensive assessment requires measuring stability, faithfulness, cross-domain consistency, and representation structure jointly. And critically, **explanation reliability is not guaranteed even when classification accuracy remains high** — domain shift can silently degrade the quality of model explanations.

---

## Future Work

- **More domains:** Extend the study to additional DomainNet domains (clipart, painting, quickdraw) to test generality of the findings across more diverse visual styles.
- **Shape-biased training:** Investigate shape-biased training augmentations (stylisation, edge-detection preprocessing) to improve real→sketch transfer without sacrificing in-domain accuracy.
- **More XAI methods:** Apply more recent XAI methods (e.g., Attention Rollout, SHAP, Score-CAM) for broader comparison.
- **Quantitative domain separation:** Develop quantitative metrics for domain separation in feature space to complement the qualitative t-SNE/UMAP visualisations.
- **Domain-adversarial training:** Investigate whether fine-tuning with domain-adversarial training can reduce the real model's domain separation and improve both transfer accuracy and explanation consistency.

---

## References

1. **DomainNet:** X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang, "Moment matching for multi-source domain adaptation," *Proc. IEEE/CVF ICCV*, 2019, pp. 1406–1415.

2. **Grad-CAM:** R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *Proc. IEEE ICCV*, 2017, pp. 618–626.

3. **Grad-CAM++:** A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks," *Proc. IEEE WACV*, 2018, pp. 839–847.

4. **Integrated Gradients:** M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks," *Proc. ICML*, vol. 70, 2017, pp. 3319–3328.

5. **LIME:** M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," *Proc. ACM SIGKDD*, 2016, pp. 1135–1144.

6. **ResNet:** K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," *Proc. IEEE CVPR*, 2016, pp. 770–778.

7. **Texture Bias:** R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel, "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness," *Proc. ICLR*, 2019.

8. **Mixup:** H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," *Proc. ICLR*, 2018.

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
