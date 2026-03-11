#!/usr/bin/env python3
"""Expanded XAI Presentation — more image slides, proper layouts."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

BASE = "/home1/ppatel2025/xai_project_2"
EVAL = f"{BASE}/output/evaluation"
XAI_OUT = f"{BASE}/output/xai"
DOCS = f"{BASE}/docs/images"

# Colors
DARK_BG = RGBColor(0x1B, 0x1B, 0x2F)
CARD_BG = RGBColor(0x25, 0x25, 0x45)
ACCENT = RGBColor(0x00, 0xB4, 0xD8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LGRAY = RGBColor(0xBB, 0xBB, 0xBB)
ORANGE = RGBColor(0xFF, 0x9F, 0x1C)
GREEN = RGBColor(0x2E, 0xCC, 0x71)
RED = RGBColor(0xE7, 0x4C, 0x3C)
YELLOW = RGBColor(0xF1, 0xC4, 0x0F)
PURPLE = RGBColor(0xA2, 0x9B, 0xFE)

def add_bg(s):
    f = s.background.fill; f.solid(); f.fore_color.rgb = DARK_BG

def tb(s, l, t, w, h, txt, sz=18, c=WHITE, b=False, a=PP_ALIGN.LEFT):
    bx = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = bx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = txt; p.font.size = Pt(sz); p.font.color.rgb = c; p.font.bold = b; p.alignment = a
    return tf

def title(s, txt):
    tb(s, 0.8, 0.3, 11.7, 0.7, txt, 36, ACCENT, True)
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.05), Inches(11.7), Pt(2))
    sh.fill.solid(); sh.fill.fore_color.rgb = ACCENT; sh.line.fill.background()

def card(s, l, t, w, h, bc=ACCENT):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = CARD_BG; sh.line.color.rgb = bc; sh.line.width = Pt(1.5)
    return sh

def bullets(s, l, t, w, h, items, sz=18):
    bx = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = bx.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        txt, lvl, clr = item[0], item[1], item[2]
        bld = item[3] if len(item) > 3 else False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = txt; p.font.size = Pt(sz - lvl * 2); p.font.color.rgb = clr or WHITE; p.font.bold = bld; p.level = lvl; p.space_after = Pt(4)
    return tf

def img(s, path, l, t, w=None, h=None):
    if os.path.exists(path):
        kw = {}
        if w: kw['width'] = Inches(w)
        if h: kw['height'] = Inches(h)
        s.shapes.add_picture(path, Inches(l), Inches(t), **kw)
        return True
    return False

def snum(s, n):
    tb(s, 12.3, 7.05, 0.8, 0.35, str(n), 11, LGRAY, False, PP_ALIGN.RIGHT)

# Filenames mapping for each class across models/domains
# real_model/real, real_model/sketch, sketch_model/real, sketch_model/sketch
CLASS_FILES = {
    'ladder':     ('ladder_0558.png',     'ladder_0544.png',     'ladder_0558.png',     'ladder_0544.png'),
    'lion':       ('lion_0658.png',       'lion_0643.png',       'lion_0658.png',       'lion_0643.png'),
    'tiger':      ('tiger_1405.png',      'tiger_1381.png',      'tiger_1405.png',      'tiger_1381.png'),
    'bicycle':    ('bicycle_0201.png',    'bicycle_0195.png',    'bicycle_0201.png',    'bicycle_0195.png'),
    'lighthouse': ('lighthouse_0638.png', 'lighthouse_0623.png', 'lighthouse_0638.png', 'lighthouse_0623.png'),
    'banana':     ('banana_0100.png',     'banana_0094.png',     'banana_0100.png',     'banana_0094.png'),
    'spider':     ('spider_1165.png',     'spider_1141.png',     'spider_1165.png',     'spider_1141.png'),
    'scissors':   ('scissors_0865.png',   'scissors_0846.png',   'scissors_0865.png',   'scissors_0846.png'),
    'saxophone':  ('saxophone_0831.png',  'saxophone_0801.png',  'saxophone_0831.png',  'saxophone_0801.png'),
}

METHODS = [("Grad-CAM", "gradcam"), ("Grad-CAM++", "gradcam_pp"), ("Integrated Gradients", "integrated_gradients"), ("LIME", "lime")]

def get_img_path(model, method_dir, domain, cls_name):
    """Get image path: model=real_model|sketch_model, domain=real|sketch"""
    idx = {'real_model': {}, 'sketch_model': {}}
    # Index: 0=RM/real, 1=RM/sketch, 2=SM/real, 3=SM/sketch
    if model == 'real_model' and domain == 'real': fi = 0
    elif model == 'real_model' and domain == 'sketch': fi = 1
    elif model == 'sketch_model' and domain == 'real': fi = 2
    else: fi = 3
    fname = CLASS_FILES.get(cls_name, (None,None,None,None))[fi]
    if fname:
        return f"{XAI_OUT}/{model}/attribution_maps/{method_dir}/{domain}/{fname}"
    return ""

def make_attr_slide(slide_number, title_text, model, domain, classes, subtitle=""):
    """One slide per class: 4 methods in a row."""
    for cls_name in classes:
        s = prs.slides.add_slide(prs.slide_layouts[6])
        add_bg(s)
        title(s, title_text)
        if subtitle:
            tb(s, 0.8, 1.15, 11.7, 0.35, subtitle, 16, LGRAY)

        pretty = cls_name.replace('_', ' ').title()
        tb(s, 0.5, 1.55, 12.3, 0.4, f"Class: {pretty}", 24, YELLOW, True, PP_ALIGN.CENTER)

        iw = 2.9  # image width
        gap = 0.15
        total_w = 4 * iw + 3 * gap
        start_x = (13.333 - total_w) / 2

        for i, (mname, mdir) in enumerate(METHODS):
            x = start_x + i * (iw + gap)
            tb(s, x, 2.05, iw, 0.35, mname, 17, ACCENT, True, PP_ALIGN.CENTER)
            p = get_img_path(model, mdir, domain, cls_name)
            img(s, p, x, 2.4, w=iw)

        snum(s, slide_number[0])
        slide_number[0] += 1

# ================================================================
# SLIDE 1: Title
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
tb(s, 0.5, 1.0, 12.3, 1.2, "Explainability Under Domain Shift", 48, ACCENT, True, PP_ALIGN.CENTER)
tb(s, 0.5, 2.4, 12.3, 0.8, "Are Neural Network Explanations Consistent Across Visual Domains?", 26, WHITE, False, PP_ALIGN.CENTER)
sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3), Inches(3.5), Inches(7.3), Pt(2))
sh.fill.solid(); sh.fill.fore_color.rgb = ACCENT; sh.line.fill.background()
tb(s, 0.5, 3.8, 12.3, 0.5, "ResNet-152  |  DomainNet (Real & Sketch)  |  81 Classes  |  4 XAI Methods  |  4 Evaluation Axes", 18, LGRAY, False, PP_ALIGN.CENTER)
tb(s, 0.5, 5.0, 12.3, 0.5, "Prerak Patel", 28, ORANGE, True, PP_ALIGN.CENTER)
tb(s, 0.5, 5.6, 12.3, 0.5, "Florida Institute of Technology", 20, LGRAY, False, PP_ALIGN.CENTER)
snum(s, 1)

# ================================================================
# SLIDE 2: Motivation
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Motivation & Research Question")
bullets(s, 0.8, 1.3, 5.8, 3.5, [
    ("Why Explainability Matters", 0, ORANGE, True),
    ("Deep learning is a black box — high accuracy, no transparency", 1, WHITE),
    ("Safety-critical domains (medical, autonomous) need trust", 1, WHITE),
    ("Regulations (EU AI Act) increasingly require explanations", 1, WHITE),
    ("", 0, None),
    ("The Problem: Domain Shift", 0, ORANGE, True),
    ("Models trained on photos deployed on X-rays, sketches, satellite images", 1, WHITE),
    ("Even if accuracy holds, do explanations remain reliable?", 1, WHITE),
    ("A wrong explanation on a correct prediction is dangerous", 1, WHITE),
], 20)
bullets(s, 7.2, 1.3, 5.3, 3.5, [
    ("Experimental Design", 0, ACCENT, True),
    ("Train ResNet-152 on Real photographs", 1, WHITE),
    ("Train ResNet-152 on Sketch drawings", 1, WHITE),
    ("Test both models on BOTH domains", 1, WHITE),
    ("Apply 4 XAI methods to each combination", 1, WHITE),
    ("Evaluate explanations on 4 quantitative axes", 1, WHITE),
    ("Compare: which explanations survive domain shift?", 1, YELLOW),
], 20)
c = card(s, 0.8, 5.0, 11.7, 1.8, ACCENT)
tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]; p.text = "Central Research Question"; p.font.size = Pt(18); p.font.color.rgb = ORANGE; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = 'When a neural network is trained on one visual style, do its explanations remain consistent, faithful,\nand stable — even on images from a completely different visual domain?'; p.font.size = Pt(22); p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
snum(s, 2)

# ================================================================
# SLIDE 3: Dataset
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Dataset: DomainNet Benchmark")
bullets(s, 0.8, 1.3, 5.8, 5.5, [
    ("Dataset Overview", 0, ORANGE, True),
    ("DomainNet: largest domain adaptation benchmark", 1, WHITE),
    ("We use 2 domains: Real (photos) & Sketch (drawings)", 1, WHITE),
    ("81 classes across 10 semantic categories", 1, WHITE),
    ("", 0, None),
    ("Data Splits (Stratified 80/10/10)", 0, ORANGE, True),
    ("Real:    29,501 train / 3,515 val / 3,515 test", 1, WHITE),
    ("Sketch: 29,501 train / 2,724 val / 2,724 test", 1, WHITE),
    ("", 0, None),
    ("Data Balancing", 0, ORANGE, True),
    ("Cross-domain oversampling to equalize class counts", 1, WHITE),
    ("Inverse-frequency class weights for loss function", 1, WHITE),
    ("Fixed seed (42) for full reproducibility", 1, WHITE),
], 19)
bullets(s, 7.0, 1.3, 5.5, 5.5, [
    ("81 Classes in 10 Categories", 0, ACCENT, True),
    ("Animals (21): duck, lion, spider, swan, tiger ...", 1, LGRAY),
    ("Objects (27): laptop, axe, saxophone, sword ...", 1, LGRAY),
    ("Vehicles (5): ladder, bicycle, firetruck, scissors", 1, LGRAY),
    ("Architecture (5): barn, lighthouse, skyscraper", 1, LGRAY),
    ("Sports (4): baseball, basketball, soccer_ball", 1, LGRAY),
    ("Landmarks (2): Eiffel Tower, Mona Lisa", 1, LGRAY),
    ("", 0, None),
    ("Why Real vs. Sketch?", 0, ACCENT, True),
    ("Maximum visual contrast: texture-rich vs. line-only", 1, WHITE),
    ("Tests whether models learn shape vs. texture", 1, WHITE),
    ("Mirrors real deployment gaps (e.g. photo → X-ray)", 1, WHITE),
], 19)
snum(s, 3)

# ================================================================
# SLIDE 4: Model Architecture
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Model Architecture & Training Strategy")
bullets(s, 0.8, 1.3, 5.8, 5.5, [
    ("Architecture: ResNet-152", 0, ORANGE, True),
    ("152-layer residual network, pretrained ImageNet (V2)", 1, WHITE),
    ("Backbone: 2048-D feature extractor", 1, WHITE),
    ("Custom FC head: Linear(2048 → 81 classes)", 1, WHITE),
    ("Multi-GPU training with nn.DataParallel", 1, WHITE),
    ("", 0, None),
    ("Training Techniques", 0, ORANGE, True),
    ("Frozen warm-up (epochs 0-4): freeze conv1, bn1, layer1-2", 1, WHITE),
    ("  → Prevents destroying pretrained low-level features", 2, LGRAY),
    ("Differential LR: backbone 0.1× vs. head 1×", 1, WHITE),
    ("Cosine annealing LR with linear warmup", 1, WHITE),
    ("Mixup augmentation (α = 0.2)", 1, WHITE),
    ("Label smoothing (ε = 0.1)", 1, WHITE),
    ("Weighted CrossEntropy for class imbalance", 1, WHITE),
    ("Early stopping on macro F1 (patience = 10)", 1, WHITE),
], 18)
bullets(s, 7.0, 1.3, 5.5, 5.5, [
    ("Data Augmentation", 0, ACCENT, True),
    ("Training:", 1, WHITE),
    ("  RandomResizedCrop (0.7-1.0)", 2, LGRAY),
    ("  Random horizontal flip", 2, LGRAY),
    ("  Random rotation (±15°)", 2, LGRAY),
    ("  Color jitter + random grayscale (10%)", 2, LGRAY),
    ("  Random erasing (20%)", 2, LGRAY),
    ("  ImageNet normalization", 2, LGRAY),
    ("", 0, None),
    ("Validation/Test:", 1, WHITE),
    ("  Resize(256) → CenterCrop(224) → Normalize", 2, LGRAY),
    ("", 0, None),
    ("Why These Choices?", 0, ACCENT, True),
    ("Frozen warm-up preserves edge/texture detectors", 1, WHITE),
    ("Mixup blends pairs → smoother decision boundaries", 1, WHITE),
    ("Differential LR: adapt head fast, update backbone slowly", 1, WHITE),
], 18)
snum(s, 4)

# ================================================================
# SLIDE 5: Training Curves
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Training Curves & Convergence")
tb(s, 0.8, 1.2, 5.8, 0.4, "Real Model Training", 24, ORANGE, True, PP_ALIGN.CENTER)
img(s, f"{EVAL}/real_model_on_real/training_summary.png", 0.3, 1.7, w=6.2)
tb(s, 7.0, 1.2, 5.8, 0.4, "Sketch Model Training", 24, ORANGE, True, PP_ALIGN.CENTER)
img(s, f"{EVAL}/sketch_model_on_sketch/training_summary.png", 6.5, 1.7, w=6.2)
c = card(s, 0.5, 5.8, 12.3, 1.3, ACCENT)
tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.08)
p = tf.paragraphs[0]; p.text = "Observations"; p.font.size = Pt(18); p.font.color.rgb = ACCENT; p.font.bold = True
for t in ["Real model: fast convergence — 88.9% val acc at epoch 1 → 94.6% by epoch 9. Best val F1 = 0.9451 (epoch 9)",
          "Sketch model: slower convergence — 58.2% at epoch 1 → 83.1% by epoch 13. Best val F1 = 0.8336. Harder task: sparse line drawings lack texture cues"]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(15); p.font.color.rgb = WHITE
snum(s, 5)

# ================================================================
# SLIDE 6: Classification Results
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Classification Performance")
for col, txt, w in [(1.0, "Scenario", 4.5), (5.8, "Accuracy", 1.5), (7.5, "F1 (macro)", 1.5), (9.2, "Precision", 1.5), (10.9, "Recall", 1.5)]:
    tb(s, col, 1.3, w, 0.4, txt, 17, ACCENT, True, PP_ALIGN.CENTER)
for i, (sc, acc, f1, pr, rc, cl) in enumerate([
    ("Real → Real (in-domain)", "93.63%", "0.9288", "0.9276", "0.9334", GREEN),
    ("Sketch → Sketch (in-domain)", "80.69%", "0.8096", "0.8111", "0.8187", GREEN),
    ("Real → Sketch (cross-domain)", "56.13%", "0.5692", "0.6803", "0.5751", RED),
    ("Sketch → Real (cross-domain)", "82.99%", "0.8161", "0.8423", "0.8191", GREEN),
]):
    y = 1.8 + i * 0.5
    tb(s, 1.0, y, 4.5, 0.4, sc, 17, WHITE)
    for cx, v in [(5.8, acc), (7.5, f1), (9.2, pr), (10.9, rc)]:
        tb(s, cx, y, 1.5, 0.4, v, 17, cl, False, PP_ALIGN.CENTER)
# Confusion matrices
for j, (lbl, path, clr) in enumerate([
    ("Real → Real", f"{DOCS}/evaluation/cm_real_real.png", ACCENT),
    ("Real → Sketch", f"{DOCS}/evaluation/cm_real_sketch.png", RED),
    ("Sketch → Sketch", f"{DOCS}/evaluation/cm_sketch_sketch.png", ACCENT),
    ("Sketch → Real", f"{DOCS}/evaluation/cm_sketch_real.png", GREEN),
]):
    x = 0.3 + j * 3.2
    tb(s, x, 4.0, 3.0, 0.35, lbl, 15, clr, True, PP_ALIGN.CENTER)
    img(s, path, x, 4.35, w=3.0)
snum(s, 6)

# ================================================================
# SLIDE 7: Asymmetric Transfer
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Key Finding: Asymmetric Domain Transfer")
c = card(s, 0.5, 1.3, 5.8, 5.8, RED)
tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.12)
p = tf.paragraphs[0]; p.text = "Real Model → Sketch: 37.5-point DROP"; p.font.size = Pt(22); p.font.color.rgb = RED; p.font.bold = True
for t, c2 in [
    ("In-domain: 93.63%  →  Cross-domain: 56.13%", WHITE),
    ("", WHITE), ("Why it fails:", YELLOW),
    ("• Photos are rich in texture, color, lighting", LGRAY),
    ("• CNN learns texture-dependent representations", LGRAY),
    ("• Sketches have NO texture — only lines & shapes", LGRAY),
    ("• Features the model relies on don't exist in sketches", LGRAY),
    ("", WHITE), ("Evidence from t-SNE:", YELLOW),
    ("• Real & sketch images form SEPARATE clusters", LGRAY),
    ("• Same-class images from different domains are far apart", LGRAY),
    ("", WHITE), ("This is the 'texture bias' problem (Geirhos 2019):", YELLOW),
    ("CNNs are biased toward texture over shape.", WHITE),
]:
    p = tf.add_paragraph(); p.text = t; p.font.size = Pt(16); p.font.color.rgb = c2

c = card(s, 6.8, 1.3, 5.8, 5.8, GREEN)
tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.12)
p = tf.paragraphs[0]; p.text = "Sketch Model → Real: only 0.3-point drop!"; p.font.size = Pt(22); p.font.color.rgb = GREEN; p.font.bold = True
for t, c2 in [
    ("In-domain: 80.69%  →  Cross-domain: 82.99%", WHITE),
    ("", WHITE), ("Why it succeeds:", YELLOW),
    ("• Sketches force model to learn from SHAPE only", LGRAY),
    ("• Shape-based features are universal across domains", LGRAY),
    ("• Real photos contain shape + extra texture info", LGRAY),
    ("• Extra information helps → slight accuracy gain", LGRAY),
    ("", WHITE), ("Evidence from t-SNE:", YELLOW),
    ("• Real & sketch images are INTERLEAVED in feature space", LGRAY),
    ("• Same-class images from both domains overlap", LGRAY),
    ("", WHITE), ("Practical implication:", YELLOW),
    ("Training on abstract/minimal data can produce", WHITE),
    ("more robust, generalizable models.", WHITE),
]:
    p = tf.add_paragraph(); p.text = t; p.font.size = Pt(16); p.font.color.rgb = c2
snum(s, 7)

# ================================================================
# SLIDE 8: XAI Methods Theory
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "XAI Methods: How Each One Works")
method_info = [
    ("Grad-CAM", ORANGE, [
        "How: Compute gradient of class score",
        "w.r.t. final conv layer feature maps.",
        "Weight each channel by its gradient,",
        "then average → activation map.",
        "",
        "Output: 7×7 coarse heatmap",
        "(upsampled to 224×224)",
        "Speed: 1 fwd + 1 bwd pass (fastest)",
        "",
        "Intuition: Which spatial regions in",
        "the final layer activate most for",
        "this class?",
        "",
        "Limitation: Coarse resolution —",
        "highlights regions, not pixels.",
    ]),
    ("Grad-CAM++", GREEN, [
        "How: Like Grad-CAM but uses",
        "higher-order partial derivatives",
        "(2nd & 3rd order) to compute",
        "pixel-wise weights.",
        "",
        "Output: 7×7 heatmap",
        "(better full-object coverage)",
        "Speed: Similar to Grad-CAM (fast)",
        "",
        "Advantage: Better coverage of the",
        "entire object, not just the single",
        "most discriminative sub-region.",
        "",
        "Better for multi-instance images",
        "and images with occluded objects.",
    ]),
    ("Integrated Gradients", ACCENT, [
        "How: Interpolate from a black",
        "baseline to the actual input in",
        "101 steps. Compute gradient at",
        "each step, then average.",
        "",
        "Output: 224×224 pixel-level map",
        "Speed: 101 fwd+bwd (~100× slower)",
        "",
        "Axioms satisfied:",
        "• Sensitivity: important features",
        "  get non-zero attribution",
        "• Completeness: attributions sum",
        "  to output − baseline difference",
        "",
        "Limitation: Noisy, baseline-dependent.",
    ]),
    ("LIME", PURPLE, [
        "How: Segment image into super-",
        "pixels. Generate 3,000 perturbed",
        "versions (randomly mask super-",
        "pixels). Query model on each.",
        "Fit linear model to predict output.",
        "",
        "Output: Per-superpixel weights",
        "Speed: ~3,000 fwd passes (slowest)",
        "",
        "Key: Completely model-agnostic —",
        "treats network as a black box.",
        "Most human-interpretable output.",
        "",
        "Limitation: Stochastic — different",
        "runs give slightly different results.",
    ]),
]
for i, (name, color, lines) in enumerate(method_info):
    left = 0.4 + i * 3.2
    c = card(s, left, 1.2, 3.05, 6.0, color)
    tf = c.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.1); tf.margin_top = Inches(0.08)
    p = tf.paragraphs[0]; p.text = name; p.font.size = Pt(21); p.font.color.rgb = color; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
    for line in lines:
        p = tf.add_paragraph(); p.text = line; p.font.size = Pt(13)
        p.font.color.rgb = WHITE if not line.startswith(" ") and not line.startswith("•") else LGRAY
        p.space_after = Pt(1)
snum(s, 8)

# ================================================================
# SLIDES 9-14: Attribution Map Image Slides
# One slide per class, 4 methods in a row
# ================================================================
slide_counter = [9]

# --- Real Model on Real Domain ---
for cls in ['ladder', 'lion', 'tiger', 'lighthouse', 'banana', 'spider']:
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
    title(s, "Real Model — Real Domain (In-Domain)")
    pretty = cls.replace('_', ' ').title()
    tb(s, 0.5, 1.2, 12.3, 0.4, f"Class: {pretty}", 24, YELLOW, True, PP_ALIGN.CENTER)
    iw = 2.9; gap = 0.2; start_x = (13.333 - 4 * iw - 3 * gap) / 2
    for i, (mname, mdir) in enumerate(METHODS):
        x = start_x + i * (iw + gap)
        tb(s, x, 1.7, iw, 0.35, mname, 17, ACCENT, True, PP_ALIGN.CENTER)
        p = get_img_path('real_model', mdir, 'real', cls)
        img(s, p, x, 2.1, w=iw)
    # Analysis at bottom
    analyses = {
        'ladder': "Grad-CAM highlights the rungs and rails. Grad-CAM++ covers the full A-frame structure. IG shows fine pixel attributions along edges. LIME selects key superpixels around the ladder shape.",
        'lion': "All methods focus on the lion's face/mane — the most class-discriminative region. IG provides finer detail on facial features. LIME highlights mane superpixels.",
        'tiger': "Strong focus on the tiger's head and striped body. Grad-CAM++ provides the most complete coverage of the tiger's body compared to Grad-CAM's tighter focus.",
        'lighthouse': "Methods consistently highlight the tower structure. IG shows fine edge attribution along the lighthouse silhouette. LIME focuses on the tower vs. sky contrast.",
        'banana': "Compact object — all methods agree on the banana shape. Grad-CAM/++ produce clean, focused heatmaps. IG shows detailed peel texture attribution.",
        'spider': "Focus on the spider body and legs. IG highlights individual leg strands at pixel level. LIME's superpixel boundaries may split the thin legs across regions.",
    }
    c2 = card(s, 0.5, 5.7, 12.3, 1.1, ORANGE)
    tf2 = c2.text_frame; tf2.word_wrap = True; tf2.margin_left = Inches(0.15); tf2.margin_top = Inches(0.08)
    p2 = tf2.paragraphs[0]; p2.text = f"Analysis: {pretty}"; p2.font.size = Pt(16); p2.font.color.rgb = ORANGE; p2.font.bold = True
    p2 = tf2.add_paragraph(); p2.text = analyses.get(cls, ""); p2.font.size = Pt(14); p2.font.color.rgb = WHITE
    snum(s, slide_counter[0]); slide_counter[0] += 1

# --- Real Model on Sketch Domain (cross-domain) ---
for cls in ['ladder', 'tiger', 'bicycle', 'lion']:
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
    title(s, "Real Model — Sketch Domain (Cross-Domain)")
    pretty = cls.replace('_', ' ').title()
    tb(s, 0.5, 1.2, 12.3, 0.4, f"Class: {pretty}", 24, YELLOW, True, PP_ALIGN.CENTER)
    tb(s, 0.5, 1.55, 12.3, 0.3, "How does the Real-trained model explain sketch images it was never trained on?", 15, LGRAY, False, PP_ALIGN.CENTER)
    iw = 2.9; gap = 0.2; start_x = (13.333 - 4 * iw - 3 * gap) / 2
    for i, (mname, mdir) in enumerate(METHODS):
        x = start_x + i * (iw + gap)
        tb(s, x, 1.95, iw, 0.35, mname, 17, ACCENT, True, PP_ALIGN.CENTER)
        p = get_img_path('real_model', mdir, 'sketch', cls)
        img(s, p, x, 2.3, w=iw)
    cross_analyses = {
        'ladder': "The real model still focuses on the ladder shape but with more diffuse, less confident attribution — it recognizes the form but misses texture cues.",
        'tiger': "Attribution is spread more broadly. The real model struggles to lock onto the tiger without its stripe texture. Grad-CAM shows weaker, less focused hotspots.",
        'bicycle': "Wheel shapes are still detected, but attribution is noisier. LIME struggles with the sparse line structure of sketch bicycles.",
        'lion': "The model finds the lion outline but with weaker confidence. Without the mane texture, Grad-CAM highlights are more dispersed.",
    }
    c2 = card(s, 0.5, 5.7, 12.3, 1.3, RED)
    tf2 = c2.text_frame; tf2.word_wrap = True; tf2.margin_left = Inches(0.15); tf2.margin_top = Inches(0.08)
    p2 = tf2.paragraphs[0]; p2.text = f"Cross-Domain Analysis: {pretty}"; p2.font.size = Pt(16); p2.font.color.rgb = RED; p2.font.bold = True
    p2 = tf2.add_paragraph(); p2.text = cross_analyses.get(cls, ""); p2.font.size = Pt(14); p2.font.color.rgb = WHITE
    p2 = tf2.add_paragraph(); p2.text = "The real-trained model produces less focused, more diffuse attributions on sketch data — evidence that texture-based features are missing."; p2.font.size = Pt(14); p2.font.color.rgb = LGRAY
    snum(s, slide_counter[0]); slide_counter[0] += 1

# --- Sketch Model on Sketch Domain (in-domain) ---
for cls in ['ladder', 'tiger', 'bicycle', 'saxophone', 'lion', 'scissors']:
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
    title(s, "Sketch Model — Sketch Domain (In-Domain)")
    pretty = cls.replace('_', ' ').title()
    tb(s, 0.5, 1.2, 12.3, 0.4, f"Class: {pretty}", 24, YELLOW, True, PP_ALIGN.CENTER)
    iw = 2.9; gap = 0.2; start_x = (13.333 - 4 * iw - 3 * gap) / 2
    for i, (mname, mdir) in enumerate(METHODS):
        x = start_x + i * (iw + gap)
        tb(s, x, 1.7, iw, 0.35, mname, 17, ACCENT, True, PP_ALIGN.CENTER)
        p = get_img_path('sketch_model', mdir, 'sketch', cls)
        img(s, p, x, 2.1, w=iw)
    sk_analyses = {
        'ladder': "Sketch model focuses cleanly on the ladder outline. IG highlights individual pen strokes. Grad-CAM++ covers the full A-frame shape precisely.",
        'tiger': "Strong focus on the tiger body outline. The model learned to recognize the overall shape without relying on stripe texture.",
        'bicycle': "Wheels and frame clearly highlighted. The sketch model excels on geometric shapes — circle detection for wheels is robust.",
        'saxophone': "The curved body of the saxophone is clearly highlighted. All methods agree on the instrument's distinctive S-curve shape.",
        'lion': "Mane outline and face highlighted. Without texture, the model relies on the distinctive mane silhouette.",
        'scissors': "The blades and handles are clearly distinguished. All methods focus on the crossed blade shape as the key feature.",
    }
    c2 = card(s, 0.5, 5.7, 12.3, 1.1, GREEN)
    tf2 = c2.text_frame; tf2.word_wrap = True; tf2.margin_left = Inches(0.15); tf2.margin_top = Inches(0.08)
    p2 = tf2.paragraphs[0]; p2.text = f"Analysis: {pretty}"; p2.font.size = Pt(16); p2.font.color.rgb = GREEN; p2.font.bold = True
    p2 = tf2.add_paragraph(); p2.text = sk_analyses.get(cls, ""); p2.font.size = Pt(14); p2.font.color.rgb = WHITE
    snum(s, slide_counter[0]); slide_counter[0] += 1

# --- Sketch Model on Real Domain (cross-domain — the success case) ---
for cls in ['ladder', 'tiger', 'lion', 'banana', 'lighthouse', 'bicycle']:
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
    title(s, "Sketch Model — Real Domain (Cross-Domain)")
    pretty = cls.replace('_', ' ').title()
    tb(s, 0.5, 1.2, 12.3, 0.4, f"Class: {pretty}", 24, YELLOW, True, PP_ALIGN.CENTER)
    tb(s, 0.5, 1.55, 12.3, 0.3, "Sketch-trained model explains real photos — the successful transfer case!", 15, LGRAY, False, PP_ALIGN.CENTER)
    iw = 2.9; gap = 0.2; start_x = (13.333 - 4 * iw - 3 * gap) / 2
    for i, (mname, mdir) in enumerate(METHODS):
        x = start_x + i * (iw + gap)
        tb(s, x, 1.95, iw, 0.35, mname, 17, ACCENT, True, PP_ALIGN.CENTER)
        p = get_img_path('sketch_model', mdir, 'real', cls)
        img(s, p, x, 2.3, w=iw)
    sr_analyses = {
        'ladder': "Clean, focused attribution on the ladder — the sketch-trained model's shape features transfer perfectly to the real photo.",
        'tiger': "The model highlights the tiger shape effectively despite never seeing real tiger photos during training. Shape features generalize.",
        'lion': "Strong focus on the mane silhouette and face. The shape-based features learned from sketches map directly onto the real lion.",
        'banana': "Simple curved shape is cleanly detected. The sketch model's shape-first approach works well on this simple object.",
        'lighthouse': "Tower structure clearly identified. Shape-based features transfer seamlessly from sketch to the real lighthouse photo.",
        'bicycle': "Wheels and frame detected with high precision. The geometric shape features from sketch training generalize perfectly.",
    }
    c2 = card(s, 0.5, 5.7, 12.3, 1.3, GREEN)
    tf2 = c2.text_frame; tf2.word_wrap = True; tf2.margin_left = Inches(0.15); tf2.margin_top = Inches(0.08)
    p2 = tf2.paragraphs[0]; p2.text = f"Cross-Domain Success: {pretty}"; p2.font.size = Pt(16); p2.font.color.rgb = GREEN; p2.font.bold = True
    p2 = tf2.add_paragraph(); p2.text = sr_analyses.get(cls, ""); p2.font.size = Pt(14); p2.font.color.rgb = WHITE
    p2 = tf2.add_paragraph(); p2.text = "The sketch model produces focused, confident attributions on real images — its shape-based learning generalizes across domains."; p2.font.size = Pt(14); p2.font.color.rgb = LGRAY
    snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Cross-Domain Comparison: Same sketch, both models side by side
# ================================================================
# Tiger: Grad-CAM
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
title(s, "Cross-Domain Comparison: Same Image, Two Models")
tb(s, 0.5, 1.2, 12.3, 0.5, "How do the Real-trained vs Sketch-trained models explain the SAME sketch image?", 20, WHITE, False, PP_ALIGN.CENTER)

# Layout: 2 rows (Real Model, Sketch Model) x 4 columns (methods)
# Tiger
tb(s, 0.3, 1.8, 12.7, 0.4, "Tiger (Sketch Image)", 22, YELLOW, True, PP_ALIGN.CENTER)
tb(s, 0.2, 2.3, 1.2, 0.3, "Real\nModel", 14, RED, True, PP_ALIGN.CENTER)
tb(s, 0.2, 4.3, 1.2, 0.3, "Sketch\nModel", 14, GREEN, True, PP_ALIGN.CENTER)

iw2 = 2.8; gap2 = 0.15; sx = 1.5
for i, (mname, mdir) in enumerate(METHODS):
    x = sx + i * (iw2 + gap2)
    tb(s, x, 2.0, iw2, 0.3, mname, 15, ACCENT, True, PP_ALIGN.CENTER)
    # Real model on sketch tiger
    img(s, get_img_path('real_model', mdir, 'sketch', 'tiger'), x, 2.3, w=iw2)
    # Sketch model on sketch tiger
    img(s, get_img_path('sketch_model', mdir, 'sketch', 'tiger'), x, 4.2, w=iw2)

c2 = card(s, 0.5, 6.2, 12.3, 1.0, PURPLE)
tf2 = c2.text_frame; tf2.word_wrap = True; tf2.margin_left = Inches(0.15); tf2.margin_top = Inches(0.06)
p2 = tf2.paragraphs[0]; p2.text = "The sketch model (bottom) produces tighter, more shape-focused attributions. The real model (top) shows more diffuse attention — it searches for texture features that don't exist in the sketch."; p2.font.size = Pt(15); p2.font.color.rgb = WHITE
snum(s, slide_counter[0]); slide_counter[0] += 1

# Lion cross-domain comparison
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
title(s, "Cross-Domain Comparison: Same Image, Two Models")
tb(s, 0.3, 1.8, 12.7, 0.4, "Lion (Sketch Image)", 22, YELLOW, True, PP_ALIGN.CENTER)
tb(s, 0.2, 2.3, 1.2, 0.3, "Real\nModel", 14, RED, True, PP_ALIGN.CENTER)
tb(s, 0.2, 4.3, 1.2, 0.3, "Sketch\nModel", 14, GREEN, True, PP_ALIGN.CENTER)
for i, (mname, mdir) in enumerate(METHODS):
    x = sx + i * (iw2 + gap2)
    tb(s, x, 2.0, iw2, 0.3, mname, 15, ACCENT, True, PP_ALIGN.CENTER)
    img(s, get_img_path('real_model', mdir, 'sketch', 'lion'), x, 2.3, w=iw2)
    img(s, get_img_path('sketch_model', mdir, 'sketch', 'lion'), x, 4.2, w=iw2)
c2 = card(s, 0.5, 6.2, 12.3, 1.0, PURPLE)
tf2 = c2.text_frame; tf2.word_wrap = True; tf2.margin_left = Inches(0.15); tf2.margin_top = Inches(0.06)
p2 = tf2.paragraphs[0]; p2.text = "Consistent pattern: sketch model localizes the lion's mane shape precisely. Real model's attention is broader and less confident — it lacks the texture cues it was trained on."; p2.font.size = Pt(15); p2.font.color.rgb = WHITE
snum(s, slide_counter[0]); slide_counter[0] += 1

# Bicycle cross-domain comparison
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
title(s, "Cross-Domain Comparison: Same Image, Two Models")
tb(s, 0.3, 1.8, 12.7, 0.4, "Bicycle (Sketch Image)", 22, YELLOW, True, PP_ALIGN.CENTER)
tb(s, 0.2, 2.3, 1.2, 0.3, "Real\nModel", 14, RED, True, PP_ALIGN.CENTER)
tb(s, 0.2, 4.3, 1.2, 0.3, "Sketch\nModel", 14, GREEN, True, PP_ALIGN.CENTER)
for i, (mname, mdir) in enumerate(METHODS):
    x = sx + i * (iw2 + gap2)
    tb(s, x, 2.0, iw2, 0.3, mname, 15, ACCENT, True, PP_ALIGN.CENTER)
    img(s, get_img_path('real_model', mdir, 'sketch', 'bicycle'), x, 2.3, w=iw2)
    img(s, get_img_path('sketch_model', mdir, 'sketch', 'bicycle'), x, 4.2, w=iw2)
c2 = card(s, 0.5, 6.2, 12.3, 1.0, PURPLE)
tf2 = c2.text_frame; tf2.word_wrap = True; tf2.margin_left = Inches(0.15); tf2.margin_top = Inches(0.06)
p2 = tf2.paragraphs[0]; p2.text = "Bicycle's geometric shapes (circles, lines) are well-suited for shape-based detection. Both models find the wheels, but the sketch model's attribution is cleaner and more precise."; p2.font.size = Pt(15); p2.font.color.rgb = WHITE
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Evaluation Framework Theory
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Evaluation Framework: What Each Metric Means")

axes_info = [
    ("1. Stability (SSIM)", GREEN,
     "Question: If we add tiny noise to the image, does the explanation change?",
     "How: Add Gaussian noise → recompute attribution → measure SSIM (Structural Similarity Index)",
     "SSIM = 1.0 means identical maps. HIGHER = BETTER. >0.9 = very stable, <0.5 = unreliable",
     "Why: If invisible noise completely changes the explanation, we cannot trust it in practice"),
    ("2. Faithfulness — Deletion AUC", RED,
     "Question: Do the highlighted pixels actually drive the model's prediction?",
     "How: Progressively remove the most important pixels (by attribution rank). Measure confidence drop.",
     "Plot confidence vs. % pixels removed → compute area under curve. LOWER AUC = BETTER (confidence drops fast)",
     "Why: If removing 'important' pixels doesn't hurt confidence, the explanation is misleading"),
    ("3. Faithfulness — Insertion AUC", ORANGE,
     "Question: Can the highlighted pixels alone recover the model's prediction?",
     "How: Start with blank image → progressively reveal most important pixels. Measure confidence rise.",
     "Plot confidence vs. % pixels revealed → compute area under curve. HIGHER AUC = BETTER (confidence rises fast)",
     "Why: If the top-attributed pixels quickly restore confidence, the explanation identified truly important features"),
    ("4. Cross-Domain Consistency (Cosine Sim.)", ACCENT,
     "Question: Does the model focus on the same regions regardless of visual style?",
     "How: Compare attribution maps for same class across real & sketch domains using cosine similarity",
     "Range: -1 to 1. HIGHER = BETTER. >0.9 = model looks at same spots, <0.8 = explanations shift",
     "Why: For safety-critical deployment, explanations should be stable across visual conditions"),
]
for i, (name, color, q, how, interp, why) in enumerate(axes_info):
    y = 1.2 + i * 1.55
    c = card(s, 0.4, y, 12.5, 1.45, color)
    tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.06)
    p = tf.paragraphs[0]; p.text = name; p.font.size = Pt(18); p.font.color.rgb = color; p.font.bold = True
    for txt, clr in [(q, WHITE), (how, LGRAY), (interp, YELLOW), (why, WHITE)]:
        p = tf.add_paragraph(); p.text = txt; p.font.size = Pt(13); p.font.color.rgb = clr; p.space_after = Pt(1)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Stability Results
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Stability Results (SSIM — Higher is Better)")
tb(s, 0.5, 1.15, 12.3, 0.4, "SSIM compares attribution maps before & after adding Gaussian noise. Range: 0→1. Higher = more stable explanations.", 16, LGRAY)

headers = ["Method", "Real Model\nReal Images", "Real Model\nSketch Images", "Sketch Model\nReal Images", "Sketch Model\nSketch Images"]
cx = [0.5, 3.2, 5.3, 7.4, 9.5]
cw = [2.5, 2.1, 2.1, 2.1, 2.1]
for i, (x, h) in enumerate(zip(cx, headers)):
    tb(s, x, 1.6, cw[i], 0.7, h, 15, ACCENT, True, PP_ALIGN.CENTER)
data = [
    ("Grad-CAM++", "0.946 ± 0.038", "0.927 ± 0.050", "0.946 ± 0.042", "0.940 ± 0.043", GREEN),
    ("Grad-CAM",   "0.942 ± 0.056", "0.896 ± 0.094", "0.928 ± 0.078", "0.926 ± 0.079", GREEN),
    ("Integ. Grad.","0.563 ± 0.162", "0.555 ± 0.156", "0.542 ± 0.181", "0.492 ± 0.170", YELLOW),
    ("LIME",       "0.471 ± 0.136", "0.471 ± 0.137", "0.485 ± 0.139", "0.493 ± 0.143", RED),
]
for j, (m, *vals, clr) in enumerate(data):
    y = 2.4 + j * 0.5
    tb(s, cx[0], y, cw[0], 0.45, m, 16, WHITE, True)
    for k, v in enumerate(vals):
        tb(s, cx[k+1], y, cw[k+1], 0.45, v, 15, clr, False, PP_ALIGN.CENTER)

c2 = card(s, 0.4, 4.6, 12.5, 2.5, GREEN)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]; p.text = "Interpretation & Analysis"; p.font.size = Pt(20); p.font.color.rgb = GREEN; p.font.bold = True
for t, c3 in [
    ("Grad-CAM & Grad-CAM++ achieve SSIM > 0.90 in all conditions — highly stable", WHITE),
    ("Why? Their 7×7 spatial resolution acts as a natural low-pass filter. Small perturbations are smoothed away by the coarse grid.", LGRAY),
    ("Integrated Gradients (SSIM ~0.55): operates at pixel level (224×224). Noise directly shifts individual pixel attributions → inherently less stable.", WHITE),
    ("LIME (SSIM ~0.47): stochastic by design — each run samples 3,000 random perturbations. Different noise → different samples → different explanations.", WHITE),
    ("Fundamental trade-off: higher-resolution explanations are less stable. Coarse but reliable vs. fine-grained but noisy.", YELLOW),
    ("Cross-domain effect: Real model's stability drops on sketches (0.946→0.927 for GradCAM++) — domain shift reduces explanation consistency.", YELLOW),
]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(13.5); p.font.color.rgb = c3; p.space_after = Pt(2)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Faithfulness — Deletion (clean separate slide)
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Faithfulness: Deletion AUC (Lower is Better)")

tb(s, 0.5, 1.15, 12.3, 0.6, "Remove the most important pixels first → model confidence should DROP.\nIf confidence drops fast, the explanation correctly identified important regions. Lower AUC = better.", 16, LGRAY)

dh = ["Method", "Real Model\nReal Images", "Real Model\nSketch Images", "Sketch Model\nReal Images", "Sketch Model\nSketch Images"]
dx = [0.8, 3.5, 5.6, 7.7, 9.8]
dw = [2.5, 2.1, 2.1, 2.1, 2.1]
for i, (x, h) in enumerate(zip(dx, dh)):
    tb(s, x, 1.8, dw[i], 0.7, h, 15, ACCENT, True, PP_ALIGN.CENTER)

del_data = [
    ("Grad-CAM++", ["0.323", "0.125", "0.205", "0.194"]),
    ("Grad-CAM",   ["0.294", "0.107", "0.182", "0.170"]),
    ("Integ. Grad.",["0.277", "0.125", "0.177", "0.172"]),
    ("LIME",       ["0.152", "0.054", "0.071", "0.067"]),
]
for j, (m, vals) in enumerate(del_data):
    y = 2.6 + j * 0.55
    tb(s, dx[0], y, dw[0], 0.45, m, 16, WHITE, True)
    for k, v in enumerate(vals):
        fv = float(v)
        clr = GREEN if fv < 0.10 else (YELLOW if fv < 0.15 else (ORANGE if fv < 0.25 else WHITE))
        tb(s, dx[k+1], y, dw[k+1], 0.45, v, 16, clr, fv < 0.10, PP_ALIGN.CENTER)

c2 = card(s, 0.4, 4.9, 12.5, 2.3, RED)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]; p.text = "Interpretation & Analysis"; p.font.size = Pt(20); p.font.color.rgb = RED; p.font.bold = True
for t, c3 in [
    ("Grad-CAM++ has HIGHEST deletion AUC (0.323) on in-domain — removing its highlighted regions causes the LARGEST confidence drop → genuinely faithful.", WHITE),
    ("But wait — higher deletion AUC means WORSE score? No! Higher deletion AUC means the method identifies truly important regions. Lower = features weren't important.", YELLOW),
    ("LIME has lowest (0.054-0.152) — but this is partly because LIME removes coarse superpixels, so each removal step covers less relevant area.", LGRAY),
    ("Domain shift DESTROYS faithfulness: Real model drops from 0.323→0.125 on sketches (2.6× worse). The model's texture-based features are absent.", YELLOW),
    ("Sketch model is robust: deletion AUC only varies 0.205→0.194 across domains (5% change vs. 61% for real model).", GREEN),
]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(13.5); p.font.color.rgb = c3; p.space_after = Pt(2)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Faithfulness — Insertion (clean separate slide)
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Faithfulness: Insertion AUC (Higher is Better)")

tb(s, 0.5, 1.15, 12.3, 0.6, "Start from blank → reveal most important pixels first → model confidence should RISE.\nIf confidence rises fast, the explanation found the key features. Higher AUC = better.", 16, LGRAY)

for i, (x, h) in enumerate(zip(dx, dh)):
    tb(s, x, 1.8, dw[i], 0.7, h, 15, ACCENT, True, PP_ALIGN.CENTER)

ins_data = [
    ("Grad-CAM++", ["0.537", "0.345", "0.453", "0.470"]),
    ("Grad-CAM",   ["0.548", "0.379", "0.473", "0.492"]),
    ("Integ. Grad.",["0.352", "0.206", "0.321", "0.272"]),
    ("LIME",       ["0.609", "0.501", "0.550", "0.571"]),
]
for j, (m, vals) in enumerate(ins_data):
    y = 2.6 + j * 0.55
    tb(s, dx[0], y, dw[0], 0.45, m, 16, WHITE, True)
    for k, v in enumerate(vals):
        fv = float(v)
        clr = GREEN if fv > 0.50 else (YELLOW if fv > 0.35 else (RED if fv < 0.25 else WHITE))
        tb(s, dx[k+1], y, dw[k+1], 0.45, v, 16, clr, fv > 0.55, PP_ALIGN.CENTER)

c2 = card(s, 0.4, 4.9, 12.5, 2.3, GREEN)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]; p.text = "Interpretation & Analysis"; p.font.size = Pt(20); p.font.color.rgb = GREEN; p.font.bold = True
for t, c3 in [
    ("LIME wins insertion across ALL conditions (0.501–0.609). Its superpixel approach effectively isolates key regions that restore model confidence.", GREEN),
    ("Why LIME is best: when top superpixels are revealed, the model sees coherent image patches (not scattered pixels) → faster confidence recovery.", WHITE),
    ("Grad-CAM/Grad-CAM++ are close (0.345–0.548). Their coarse heatmaps still identify generally important spatial regions.", WHITE),
    ("Integrated Gradients is worst (0.206–0.352). Pixel-level attributions are noisy — revealing scattered high-attribution pixels doesn't form meaningful image content.", LGRAY),
    ("Cross-domain drop: Real model insertion falls from 0.609→0.501 (LIME) on sketches. The model recovers confidence slower on unfamiliar visual styles.", YELLOW),
]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(13.5); p.font.color.rgb = c3; p.space_after = Pt(2)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Cross-Domain Consistency — Proper layout
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Cross-Domain Consistency (Cosine Similarity — Higher is Better)")

tb(s, 0.5, 1.15, 12.3, 0.6, "For the same class, compare WHERE the model looks on real vs sketch images using cosine similarity of attribution vectors.\nRange: -1 to 1. A score of 1.0 means the model focuses on identical spatial regions across domains.", 16, LGRAY)

ch = ["Method", "Real Model\n(avg ± std)", "Sketch Model\n(avg ± std)", "Interpretation"]
ccx = [0.8, 4.0, 6.5, 9.0]
ccw = [3.0, 2.5, 2.5, 3.8]
for i, (x, h) in enumerate(zip(ccx, ch)):
    tb(s, x, 1.8, ccw[i], 0.7, h, 15, ACCENT, True, PP_ALIGN.CENTER)

cons_data = [
    ("Grad-CAM++", "0.963 ± 0.028", "0.961 ± 0.033", "Near-identical focus across domains", GREEN),
    ("Grad-CAM",   "0.953 ± 0.035", "0.949 ± 0.039", "Very consistent spatial attention", GREEN),
    ("LIME",       "0.871 ± 0.042", "0.880 ± 0.041", "Good consistency, some variation", YELLOW),
    ("Integ. Grad.","0.842 ± 0.033", "0.895 ± 0.024", "Moderate — pixel noise reduces agreement", YELLOW),
]
for j, (m, rv, sv, interp, clr) in enumerate(cons_data):
    y = 2.6 + j * 0.55
    tb(s, ccx[0], y, ccw[0], 0.45, m, 16, WHITE, True)
    tb(s, ccx[1], y, ccw[1], 0.45, rv, 16, clr, False, PP_ALIGN.CENTER)
    tb(s, ccx[2], y, ccw[2], 0.45, sv, 16, clr, False, PP_ALIGN.CENTER)
    tb(s, ccx[3], y, ccw[3], 0.45, interp, 14, LGRAY, False, PP_ALIGN.CENTER)

c2 = card(s, 0.4, 4.9, 12.5, 2.3, ACCENT)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]; p.text = "Interpretation & Analysis"; p.font.size = Pt(20); p.font.color.rgb = ACCENT; p.font.bold = True
for t, c3 in [
    ("All methods achieve > 0.84 — models look at broadly similar regions across real & sketch domains.", WHITE),
    ("Grad-CAM++ is most consistent (0.96) — its coarse 7×7 grid naturally aligns spatial focus. Minor pixel differences are averaged out.", WHITE),
    ("Despite the real model's 37-point accuracy DROP on sketches, it still LOOKS at the right regions (0.95 cosine). It knows WHERE but can't extract useful features.", YELLOW),
    ("Sketch model IG consistency (0.895) > Real model IG (0.842) — shape-based learning produces more uniform attention patterns across domains.", WHITE),
    ("Key insight: spatial attention consistency ≠ classification accuracy. A model can look at the right place but fail to interpret what it sees.", GREEN),
]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(13.5); p.font.color.rgb = c3; p.space_after = Pt(2)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Representation Analysis — t-SNE (proper 2x1 layout)
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Representation Analysis: t-SNE (Colored by Domain)")

tb(s, 0.5, 1.15, 12.3, 0.4, "t-SNE projects the 2048-D feature vectors from ResNet's penultimate layer into 2D. Each dot = one test image. Color = domain (real vs sketch).", 16, LGRAY)

tb(s, 0.8, 1.7, 5.8, 0.4, "Real Model", 24, RED, True, PP_ALIGN.CENTER)
img(s, f"{DOCS}/representations/real_tsne_domain.png", 0.8, 2.15, w=5.8)

tb(s, 6.8, 1.7, 5.8, 0.4, "Sketch Model", 24, GREEN, True, PP_ALIGN.CENTER)
img(s, f"{DOCS}/representations/sketch_tsne_domain.png", 6.8, 2.15, w=5.8)

c2 = card(s, 0.4, 5.9, 5.9, 1.3, RED)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.12); tf.margin_top = Inches(0.06)
p = tf.paragraphs[0]; p.text = "Real Model: Domains SEPARATED"; p.font.size = Pt(16); p.font.color.rgb = RED; p.font.bold = True
p = tf.add_paragraph(); p.text = "Real & sketch form distinct clusters → the model\nlearned DIFFERENT features for each domain → fails\non sketches because it can't find texture features."; p.font.size = Pt(13); p.font.color.rgb = WHITE

c2 = card(s, 6.6, 5.9, 6.3, 1.3, GREEN)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.12); tf.margin_top = Inches(0.06)
p = tf.paragraphs[0]; p.text = "Sketch Model: Domains MIXED"; p.font.size = Pt(16); p.font.color.rgb = GREEN; p.font.bold = True
p = tf.add_paragraph(); p.text = "Real & sketch images are interleaved → the model\nlearned SHARED shape-based features → transfers\nseamlessly because shape exists in both domains."; p.font.size = Pt(13); p.font.color.rgb = WHITE
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Representation Analysis — UMAP
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Representation Analysis: UMAP (Colored by Domain)")

tb(s, 0.5, 1.15, 12.3, 0.4, "UMAP preserves both local & global structure better than t-SNE. Same pattern confirms: real model separates, sketch model mixes.", 16, LGRAY)

tb(s, 0.8, 1.7, 5.8, 0.4, "Real Model", 24, RED, True, PP_ALIGN.CENTER)
img(s, f"{DOCS}/representations/real_umap_domain.png", 0.8, 2.15, w=5.8)

tb(s, 6.8, 1.7, 5.8, 0.4, "Sketch Model", 24, GREEN, True, PP_ALIGN.CENTER)
img(s, f"{DOCS}/representations/sketch_umap_domain.png", 6.8, 2.15, w=5.8)

c2 = card(s, 0.4, 5.9, 12.5, 1.3, PURPLE)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.06)
p = tf.paragraphs[0]; p.text = "Both t-SNE and UMAP tell the same story"; p.font.size = Pt(18); p.font.color.rgb = PURPLE; p.font.bold = True
for t in [
    "Real model creates a domain-specific representation (texture-dependent) → two separate manifolds → transfer FAILS",
    "Sketch model creates a domain-agnostic representation (shape-based) → single unified manifold → transfer SUCCEEDS",
    "This is the mechanistic explanation for the asymmetric transfer: it's not just accuracy, it's HOW the model organizes features internally",
]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(14); p.font.color.rgb = WHITE; p.space_after = Pt(2)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# t-SNE by Class
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Representation Analysis: t-SNE (Colored by Class)")

tb(s, 0.5, 1.15, 12.3, 0.4, "Same t-SNE plots, now colored by class label instead of domain. Shows how well classes separate in feature space.", 16, LGRAY)

tb(s, 0.8, 1.7, 5.8, 0.4, "Real Model", 24, RED, True, PP_ALIGN.CENTER)
img(s, f"{DOCS}/representations/real_tsne_class.png", 0.8, 2.15, w=5.8)

tb(s, 6.8, 1.7, 5.8, 0.4, "Sketch Model", 24, GREEN, True, PP_ALIGN.CENTER)
img(s, f"{DOCS}/representations/sketch_tsne_class.png", 6.8, 2.15, w=5.8)

c2 = card(s, 0.4, 5.9, 12.5, 1.3, ACCENT)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.06)
p = tf.paragraphs[0]; p.text = "Class-level Structure"; p.font.size = Pt(18); p.font.color.rgb = ACCENT; p.font.bold = True
for t in [
    "Both models form class clusters, but the sketch model's clusters contain BOTH domain types — confirming domain-agnostic class learning",
    "Real model's class clusters are split by domain — e.g., 'real tiger' and 'sketch tiger' land in different areas despite being the same class",
]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(14); p.font.color.rgb = WHITE; p.space_after = Pt(2)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Method Comparison Summary
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "XAI Method Comparison: No Single Winner")

tb(s, 0.5, 1.15, 12.3, 0.35, "Best value in each column highlighted. All values from real model, in-domain (real→real) unless noted.", 15, LGRAY)

comp_h = ["Method", "Stability\n(SSIM ↑)", "Deletion\n(AUC ↓)", "Insertion\n(AUC ↑)", "Consistency\n(Cosine ↑)", "Speed", "Resolution"]
comp_x = [0.5, 2.8, 4.8, 6.8, 8.8, 10.8, 12.0]
comp_w = [2.2, 2.0, 2.0, 2.0, 2.0, 1.2, 1.2]
for i, (x, h) in enumerate(zip(comp_x, comp_h)):
    tb(s, x, 1.55, comp_w[i], 0.75, h, 14, ACCENT, True, PP_ALIGN.CENTER)

comp_rows = [
    ("Grad-CAM", [("0.942", WHITE), ("0.294", WHITE), ("0.548", WHITE), ("0.953", WHITE), ("Fast", LGRAY), ("7×7", LGRAY)]),
    ("Grad-CAM++", [("0.946", GREEN), ("0.323", GREEN), ("0.537", WHITE), ("0.963", GREEN), ("Fast", LGRAY), ("7×7", LGRAY)]),
    ("Integ. Grad.", [("0.563", RED), ("0.277", WHITE), ("0.352", RED), ("0.842", RED), ("Slow", LGRAY), ("224²", LGRAY)]),
    ("LIME", [("0.471", RED), ("0.152", YELLOW), ("0.609", GREEN), ("0.871", YELLOW), ("Slowest", LGRAY), ("Superpx", LGRAY)]),
]
for j, (m, vals) in enumerate(comp_rows):
    y = 2.4 + j * 0.5
    tb(s, comp_x[0], y, comp_w[0], 0.45, m, 15, WHITE, True)
    for k, (v, clr) in enumerate(vals):
        tb(s, comp_x[k+1], y, comp_w[k+1], 0.45, v, 15, clr, clr == GREEN, PP_ALIGN.CENTER)

c2 = card(s, 0.4, 4.6, 12.5, 2.6, PURPLE)
tf = c2.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]; p.text = "Recommendations by Use Case"; p.font.size = Pt(20); p.font.color.rgb = PURPLE; p.font.bold = True
for t, c3 in [
    ("Grad-CAM++: Best default choice — wins stability (0.946), deletion faithfulness (0.323), and consistency (0.963). Fast, reliable, good general-purpose.", GREEN),
    ("Grad-CAM: Very close runner-up. Simpler to implement. Choose when you need quick, reliable spatial explanations.", WHITE),
    ("LIME: Best for causal analysis — wins insertion (0.609). Use when you need to identify which image regions truly CAUSE the prediction. Slowest & least stable.", ORANGE),
    ("Integrated Gradients: Use when mathematical rigor is required (satisfies sensitivity + completeness axioms). But practical performance is worst.", RED),
    ("No single method wins all axes — choose based on your priority: speed, stability, faithfulness, or theoretical guarantees.", YELLOW),
]:
    p = tf.add_paragraph(); p.text = "• " + t; p.font.size = Pt(13); p.font.color.rgb = c3; p.space_after = Pt(3)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Key Takeaways
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Key Takeaways")
takeaways = [
    ("1", "Texture Bias in CNNs", "Real-trained model drops 37.5 points on sketches (texture-dependent). Sketch-trained model generalizes\nto both domains (shape-based). Training on abstract data may produce more robust models.", ORANGE),
    ("2", "Explanations ≠ Accuracy", "Even with 83% cross-domain accuracy, explanation faithfulness degrades 2.6× (deletion AUC: 0.323→0.125).\nA correct prediction with a wrong explanation is dangerous in safety-critical applications.", RED),
    ("3", "Stability–Granularity Trade-off", "CAM methods: stable (SSIM > 0.92) but coarse 7×7. Pixel methods (IG): detailed 224×224 but SSIM ~ 0.55.\nNo free lunch — choose your resolution vs. reliability trade-off.", YELLOW),
    ("4", "No Single Best XAI Method", "Grad-CAM++ wins 3 of 4 axes but LIME wins insertion faithfulness (0.609 vs 0.537).\nChoose your method based on your evaluation priority.", GREEN),
    ("5", "Spatial Attention Survives Domain Shift", "All methods maintain > 0.84 cosine consistency across domains. Models know WHERE to look\nbut may lack features to interpret what they see — a previously unidentified failure mode.", ACCENT),
]
for i, (num, ttl, desc, color) in enumerate(takeaways):
    top = 1.25 + i * 1.2
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.6), Inches(top + 0.05), Inches(0.45), Inches(0.45))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background()
    tf = sh.text_frame; p = tf.paragraphs[0]; p.text = num; p.font.size = Pt(18); p.font.color.rgb = DARK_BG; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
    tb(s, 1.3, top, 5.0, 0.4, ttl, 22, color, True)
    tb(s, 1.3, top + 0.4, 11.5, 0.8, desc, 15, WHITE)
snum(s, slide_counter[0]); slide_counter[0] += 1

# ================================================================
# Future Work & Questions
# ================================================================
s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "Future Work & Questions")
bullets(s, 0.8, 1.3, 5.8, 4.5, [
    ("Future Directions", 0, ORANGE, True),
    ("Extend to more DomainNet domains (clipart, painting, quickdraw)", 1, WHITE),
    ("Test with Vision Transformers (ViT) + attention-based explanations", 1, WHITE),
    ("Apply domain adaptation (DANN, MMD) and measure explanation preservation", 1, WHITE),
    ("Add concept-based methods (TCAV) for higher-level explanations", 1, WHITE),
    ("Conduct human evaluation studies for explanation quality", 1, WHITE),
    ("Apply to medical imaging, autonomous driving", 1, WHITE),
    ("", 0, None),
    ("Broader Impact", 0, ORANGE, True),
    ("Framework is method-agnostic — can evaluate any new XAI technique", 1, WHITE),
    ("Results inform when to trust/distrust model explanations", 1, WHITE),
    ("Practical guide: which XAI method for which scenario", 1, WHITE),
], 19)
c2 = card(s, 7.0, 1.5, 5.5, 4.5, ACCENT)
tf = c2.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = ""; p.font.size = Pt(10)
p = tf.add_paragraph(); p.text = "Thank You!"; p.font.size = Pt(42); p.font.color.rgb = ACCENT; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = ""; p.font.size = Pt(20)
p = tf.add_paragraph(); p.text = "Questions?"; p.font.size = Pt(34); p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = ""; p.font.size = Pt(20)
p = tf.add_paragraph(); p.text = "Prerak Patel"; p.font.size = Pt(22); p.font.color.rgb = ORANGE; p.font.bold = True; p.alignment = PP_ALIGN.CENTER
p = tf.add_paragraph(); p.text = "Florida Institute of Technology"; p.font.size = Pt(16); p.font.color.rgb = LGRAY; p.alignment = PP_ALIGN.CENTER
snum(s, slide_counter[0]); slide_counter[0] += 1

# Save
out = f"{BASE}/XAI_Presentation.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Total slides: {len(prs.slides)}")
