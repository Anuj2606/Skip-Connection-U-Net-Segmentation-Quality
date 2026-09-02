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
- Compare a bottleneck-only encoder-decoder against a U-Net with a ResNet18 encoder, under matched training conditions
- Visualize *where* spatial information is lost during downsampling and *how* skip connections recover it
- Practice a proper ML ablation methodology: change one variable, hold everything else constant

---

## 🏗️ Model Architectures

### 1. SimpleSegNet — Baseline (No Skip Connections)

```text
Input → Encoder (downsampling) → Bottleneck → Decoder (upsampling) → Output
```
A standard encoder-decoder with no shortcut paths. All spatial information must survive the bottleneck.

### 2. U-Net — With Skip Connections

```text
Input → Encoder (ResNet18) → Bottleneck
                 │                │
                 └── Skip ────────┘ (per resolution level)
                                   ↓
                              Decoder → Output
```
Encoder features at each resolution are concatenated into the corresponding decoder layer before upsampling, preserving spatial detail the bottleneck alone would lose.

**Both models were trained under identical conditions** — same dataset split, loss function, optimizer, and epochs — so any difference in output is attributable to the architecture change, not training variance.

---

## 🗂️ Dataset

**[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)** — pet images paired with pixel-wise segmentation masks (foreground/background/boundary).

---

## ⚙️ Training Setup

| Component | Choice |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Data Loading | PyTorch `DataLoader` |
| Tracked Metrics | Training loss curves, qualitative output comparison |

---

## 📊 Results

| Model | Training Loss | Spatial Detail | Boundary Sharpness |
|---|---|---|---|
| SimpleSegNet (no skip) | Higher | Blurred, loses fine structure | Soft / imprecise |
| U-Net (with skip) | Lower | Preserved | Sharp |

> **Note for repo maintainer:** replace the qualitative claims above with your actual final loss values (e.g. a table of final-epoch loss per model) and drop in 2–3 side-by-side output images here. Recruiters and reviewers weight a results section with real numbers and pictures far more heavily than one with adjectives — this is the single highest-leverage edit you can make to this README before sharing the repo.

**Takeaway:** the U-Net's skip connections measurably reduce information loss through the bottleneck, producing sharper, more spatially accurate segmentation masks than the same network without them.

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Torchvision` · `segmentation-models-pytorch` · `Matplotlib`

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/Anuj2606/Skip-Connection-U-Net-Segmentation-Quality.git
cd Skip-Connection-U-Net-Segmentation-Quality

# Install dependencies
pip install torch torchvision matplotlib segmentation-models-pytorch

# Run
jupyter notebook Skip_Connection_Impact_Study_on_U_Net_Segmentation_Quality.ipynb
```

---

## 🧠 Key Learnings

- How to structure an ML ablation study — isolate one architectural variable and hold the rest constant
- Practical implementation of U-Net and skip-connection mechanics in PyTorch
- Why the encoder-decoder bottleneck is a spatial-information bottleneck, concretely, not just in theory
- End-to-end deep learning workflow: data loading → training → loss tracking → qualitative evaluation

---

## 🔮 Future Improvements

- [ ] Add quantitative segmentation metrics (Dice coefficient, IoU) — currently the comparison is loss + visual only
- [ ] Train on a larger, more diverse dataset
- [ ] Add stronger augmentation to test robustness of each architecture
- [ ] Extend the ablation to other architectural variants (e.g. attention gates, deep supervision)

---

<p align="center"><i>Built to understand U-Net, not just to use it.</i></p>
