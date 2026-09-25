<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Segmentation-4285F4?style=for-the-badge" alt="Computer Vision" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

<h1 align="center">🧩 Skip Connections in U-Net: An Ablation Study on Segmentation Quality</h1>

<p align="center">
  <b>A controlled comparison of a plain encoder-decoder network against a U-Net with skip connections, to isolate exactly what skip connections contribute to segmentation quality.</b>
</p>

---

## 📌 Problem Statement

Encoder-decoder networks compress an image down to a low-resolution feature map and then upsample it back to full size. That compression step is lossy — fine spatial detail (edges, thin structures, boundaries) gets destroyed on the way down and can't be recovered on the way up from a bottleneck alone.

U-Net's answer is the **skip connection**: it routes high-resolution features from the encoder directly to the matching decoder layer, bypassing the bottleneck entirely. This project doesn't just implement that idea — it **isolates and measures it**, by training an identical architecture with and without skip connections and comparing the outputs directly.

---

## 🎯 Objectives

- Quantify what skip connections actually buy you in segmentation quality — not just cite that they help
- Compare a bottleneck-only encoder-decoder against a compact convolutional U-Net, under matched training conditions
- Visualize *where* spatial information is lost during downsampling and *how* skip connections recover it
- Practice a proper ML ablation methodology: change one variable, hold everything else constant

---

## 🧠 Core Concepts

**Semantic segmentation** assigns a class label to every pixel in an image. In this project, each pixel is classified as background, foreground, or boundary so the model can produce a complete pet segmentation mask.

**U-Net** is an encoder-decoder segmentation architecture. The encoder downsamples the image to learn deeper features, while the decoder upsamples those features to reconstruct a pixel-level prediction.

**Skip connections** link encoder feature maps to decoder layers at the same spatial resolution. They give the decoder access to fine details such as edges and object boundaries that may be lost in the bottleneck. The project tests their contribution by comparing the U-Net with a baseline encoder-decoder that has no such connections.

**Loss functions and metrics:** The project uses the following measures:

- **Cross-Entropy Loss:** Measures pixel-level classification error. It increases when the model assigns the wrong class to a pixel. Lower values are better.
- **Dice Score:** Measures the overlap between the predicted mask and the ground-truth mask. It ranges from 0 to 1, where 1 means a perfect overlap. Higher values are better.
- **Dice Loss:** Used during training and calculated as `1 - Dice Score`. It encourages the model to improve the overlap between predicted and target regions. Lower values are better.
- **Mean IoU:** Mean Intersection over Union measures the intersection between the predicted and target regions divided by their union. Higher values indicate better segmentation quality.

Cross-Entropy focuses on classifying each pixel correctly, while Dice Loss focuses on the overlap of the complete segmentation region. The combined Dice + Cross-Entropy loss uses both objectives.

---

## 🏗️ Model Architectures

### 1. SimpleEncoderDecoder — Baseline (No Skip Connections)

```text
Input → Encoder (downsampling) → Bottleneck → Decoder (upsampling) → Output
```
A standard encoder-decoder with no shortcut paths. The encoder uses 3→64 and 64→128 convolutional blocks with ReLU activations and 2×2 max pooling. The decoder uses transposed convolutions to upsample from the bottleneck to three output classes. All spatial information must survive the bottleneck.

### 2. U-Net — With Skip Connections

```text
Input → Encoder → Bottleneck
                 │                │
                 └── Skip ────────┘ (per resolution level)
                                   ↓
                              Decoder → Output
```
This is a compact U-Net implemented directly in PyTorch. Two convolutional encoder blocks expand the channels from 3→64→128, followed by a 256-channel bottleneck. Transposed-convolution decoder blocks concatenate encoder features at matching resolutions before convolution, preserving spatial detail the bottleneck alone would lose. Both models produce logits for the same three segmentation classes.

**Both models were trained under identical conditions** — same dataset split, loss function, optimizer, and epochs — so any difference in output is attributable to the architecture change, not training variance.

---

## 🗂️ Dataset

**[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)** — pet images paired with pixel-wise segmentation masks (foreground/background/boundary).

The notebook uses the `trainval` split for training and the `test` split for evaluation. Images are resized to `128 × 128` and converted to tensors. Segmentation masks use nearest-neighbor resizing to preserve class labels, are converted from PIL images to tensors, and are remapped from the dataset values `{1, 2, 3}` to zero-indexed class labels `{0, 1, 2}`.

---

## ⚙️ Training Setup

| Component | Choice |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (`lr=1e-3`) |
| Batch Size | 16 for training and initial evaluation |
| Training Device | CUDA GPU when available; Google Colab NVIDIA T4 in the notebook |
| Data Loading | PyTorch `DataLoader` |
| Tracked Metrics | Training loss, Mean IoU, Dice score, and qualitative output comparison |

The baseline and initial U-Net comparison run for 3 epochs with Cross-Entropy loss. The notebook then defines a `DiceCELoss` that combines Cross-Entropy with a soft Dice loss and trains a separate U-Net model for 5 epochs. A parallel evaluation loader uses batch size 32, two workers, and pinned memory.

---

## 📊 Results

| Model | Final Training Loss | Mean IoU | Dice Score |
|---|---|---|---|
| SimpleEncoderDecoder (no skip) | 0.7461 | 0.3441 | 0.4451 |
| U-Net (with skip, Cross-Entropy) | 0.6224 | 0.4879 | 0.6173 |
| U-Net (with skip, Dice + Cross-Entropy) | 0.9615 | 0.5592 | 0.6890 |

The initial architecture comparison used Cross-Entropy loss for 3 epochs. The separate Dice + Cross-Entropy experiment trained a U-Net for 5 epochs, with the combined loss decreasing from `1.4266` to `0.9615`. In the recorded evaluation, this configuration achieved the strongest Mean IoU and Dice Score of the three runs.

The notebook generates qualitative 3×4 comparison figures containing the input image, ground-truth mask, SimpleEncoderDecoder prediction, and U-Net prediction for each of three test examples.

**Takeaway:** in the recorded evaluation, adding skip connections improved Mean IoU from `0.3441` to `0.4879` and Dice score from `0.4451` to `0.6173` over the no-skip baseline. Adding Dice loss to Cross-Entropy improved the separately trained U-Net result further to `0.5592` Mean IoU and `0.6890` Dice Score. These results support the role of skip connections in preserving spatial information and show the benefit of an overlap-aware loss in this experiment.

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Torchvision` · `Matplotlib`

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/Anuj2606/Skip-Connection-U-Net-Segmentation-Quality.git
cd Skip-Connection-U-Net-Segmentation-Quality

# Install dependencies
pip install torch torchvision matplotlib

# Run
jupyter notebook Skip_Connections_in_U_Net_A_Segmentation_Ablation_Study.ipynb
```

---

## 🧠 Key Learnings

- How to structure an ML ablation study — isolate one architectural variable and hold the rest constant
- Practical implementation of U-Net and skip-connection mechanics in PyTorch
- Why the encoder-decoder bottleneck is a spatial-information bottleneck, concretely, not just in theory
- End-to-end deep learning workflow: data loading → mask preprocessing → training → loss tracking → IoU/Dice evaluation → qualitative evaluation
- How combining Cross-Entropy with Dice loss can align optimization more closely with segmentation overlap quality

---

## 🔮 Future Improvements

- [ ] Add per-class and validation-set reporting for quantitative segmentation metrics
- [ ] Train on a larger, more diverse dataset
- [ ] Add stronger augmentation to test robustness of each architecture
- [ ] Extend the ablation to other architectural variants (e.g. attention gates, deep supervision)

---

<p align="center"><i>Built to understand U-Net, not just to use it.</i></p>
