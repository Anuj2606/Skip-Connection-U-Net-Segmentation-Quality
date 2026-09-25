<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Segmentation-4285F4?style=for-the-badge" alt="Computer Vision" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

<h1 align="center">🧩 Skip Connections in U-Net: An Ablation Study on Segmentation Quality</h1>

<p align="center">
  <b>A comparative study of a baseline encoder-decoder and U-Net with skip connections for semantic segmentation, with an additional evaluation of Cross-Entropy vs Dice + Cross-Entropy loss.</b>
</p>

---

## 📌 Problem Statement

Encoder-decoder networks compress an image down to a low-resolution feature map and then upsample it back to full size. That compression step is lossy — fine spatial detail (edges, thin structures, boundaries) gets destroyed on the way down and can't be recovered on the way up from a bottleneck alone.

U-Net's answer is the **skip connection**: it routes high-resolution features from the encoder directly to the matching decoder layer, bypassing the bottleneck entirely. This project compares a baseline encoder-decoder with a compact U-Net to evaluate how skip connections affect segmentation performance under the same dataset and training setup.

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

The baseline and U-Net models were evaluated using the same dataset, preprocessing, optimizer, and training setup, with the architecture change being the main comparison point.

---

## 🗂️ Dataset

**[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)** — pet images paired with pixel-wise segmentation masks (foreground/background/boundary).

The notebook uses the `trainval` split for training and the `test` split for evaluation. Images are resized to `128 × 128` and converted to tensors. Segmentation masks use nearest-neighbor resizing to preserve class labels, are converted from PIL images to tensors, and are remapped from the dataset values `{1, 2, 3}` to zero-indexed class labels `{0, 1, 2}`.

---

## ⚙️ Training Setup

| Component | Choice |
|---|---|
| Dataset | Oxford-IIIT Pet Dataset |
| Image Size | 128 × 128 |
| Batch Size | 16 |
| Number of Classes | 3 |
| Optimizer | Adam (`lr=1e-3`) |
| Maximum Epochs | 30 |
| Early Stopping Patience | 5 |
| Training Device | CUDA when available |
| Metrics | Mean IoU and Dice Score |
| Losses | Cross-Entropy and Dice + Cross-Entropy |

The final Colab implementation is a compact convolutional U-Net implemented directly in PyTorch. It is not a ResNet18-based U-Net. Training and validation loss curves and prediction visualizations were used for qualitative analysis alongside the reported Mean IoU and Dice Score metrics.

---

## 📊 Results

| Model | Mean IoU | Dice Score |
|---|---:|---:|
| Baseline Encoder-Decoder | 0.4487 | 0.5902 |
| U-Net + Cross-Entropy | 0.6538 | 0.7756 |
| U-Net + Dice + Cross-Entropy | 0.6564 | 0.7787 |

**Observation:** The U-Net achieved substantially higher segmentation performance than the baseline encoder-decoder. Adding Dice Loss to Cross-Entropy provided a small additional improvement.

- **U-Net vs Baseline Mean IoU:** 45.73% improvement
- **U-Net vs Baseline Dice Score:** 31.43% improvement
- **Dice + Cross-Entropy vs Cross-Entropy Mean IoU:** 0.40% improvement
- **Dice + Cross-Entropy vs Cross-Entropy Dice Score:** 0.40% improvement

For the loss comparison, the absolute differences were **+0.0026 Mean IoU** and **+0.0031 Dice Score**.

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

- U-Net and skip-connection implementation in PyTorch
- Preservation of spatial information in segmentation models
- Mean IoU and Dice evaluation for model comparison
- Loss-function comparison between Cross-Entropy and Dice + Cross-Entropy
- End-to-end segmentation workflow from data prep to evaluation

---

## 🔮 Future Improvements

- [ ] Add per-class and validation-set reporting for quantitative segmentation metrics
- [ ] Train on a larger, more diverse dataset
- [ ] Add stronger augmentation to test robustness of each architecture
- [ ] Extend the ablation to other architectural variants (e.g. attention gates, deep supervision)

---

<p align="center"><i>Built to understand U-Net, not just to use it.</i></p>
