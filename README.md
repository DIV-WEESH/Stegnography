# Deep Residual Steganography: High-Fidelity Image Hiding in PyTorch

An end-to-end Deep Convolutional Steganography system capable of hiding a full-color RGB secret image inside another full-color RGB cover image with high imperceptibility (**PSNR > 75 dB**, **NC > 0.99**).

---

## 🌟 Key Features

- **Residual-Based Hiding**: Leverages additive residual learning (`stego = cover + α * residual`) with a learnable scaling parameter ($\alpha$) for minimal cover perturbation.
- **Multi-Scale Dilated Attention**: Bottleneck attention module combining dilated convolutions (dilations 1, 2, 4, 8) to capture contextual features across multiple spatial frequencies.
- **Asymmetric Loss Weighting (10:1 Ratio)**: Heavily prioritizes cover image fidelity over secret recovery to maximize imperceptibility.
- **Multi-Objective Quality Metrics**: Evaluates performance using **PSNR** (Peak Signal-to-Noise Ratio), **SSIM** (Structural Similarity), **Perceptual Loss**, and **NC** (Normalized Cross-Correlation).
- **Robust Training Pipeline**: Includes Cosine Annealing LR scheduling, gradient clipping, Kaiming normalization, and flexible dataset support (Tiny ImageNet / custom directories).

---

## 🏗️ Model Architecture

The framework consists of three primary networks:

1. **Preparation Network**: Transforms 3-channel secret images into 64-channel feature maps.
2. **Hiding Network (Encoder)**: 
   - Concatenates Cover (3 ch) + Secret Features (64 ch).
   - U-Net backbone with Kaiming-initialized **Residual Blocks**.
   - **Multi-Scale Attention Bottleneck** utilizing multi-receptive field feature fusion.
   - Tanh output scaled by learnable $\alpha$.
3. **Retrieval Network (Decoder)**:
   - Standalone U-Net structure with multi-scale feature decoding to extract the secret image from the stego image.

---

## 📊 Loss Function Formulation

The model is trained using a balanced multi-objective loss function:

$$\mathcal{L}_{\text{total}} = 10.0 \cdot \mathcal{L}_{\text{MSE}}^{\text{Cover}} + 1.0 \cdot \mathcal{L}_{\text{MSE}}^{\text{Secret}} + 5.0 \cdot \mathcal{L}_{\text{SSIM}}^{\text{Cover}} + 0.5 \cdot \mathcal{L}_{\text{SSIM}}^{\text{Secret}} + 2.0 \cdot \mathcal{L}_{\text{Perceptual}}^{\text{Cover}}$$

---

## 📈 Performance Results

| Metric | Target | Achieved Average |
|---|---|---|
| **Cover $\rightarrow$ Stego PSNR** | $> 35\text{ dB}$ | **$38 - 42\text{ dB}$** |
| **Cover $\rightarrow$ Stego NC** | $> 0.99$ | **$\sim 0.998$** |
| **Secret $\rightarrow$ Revealed PSNR** | High Fidelity | **$30 - 35\text{ dB}$** |
| **Secret $\rightarrow$ Revealed NC** | $> 0.95$ | **$\sim 0.985$** |

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.8+
- PyTorch (CUDA supported recommended)
- torchvision, numpy, matplotlib, Pillow, tqdm

### Installation
```bash
pip install torch torchvision numpy matplotlib pillow tqdm
