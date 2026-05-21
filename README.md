# GradAttn: Transformer Based Modulation of Residual Approach for Classification and Representation Learning Problems

**Soudeep Ghoshal** (Kalinga Institute of Industrial Technology) and **Himanshu Buckchash** (IMC University of Applied Sciences Krems)

---

## About

This repository contains the experimental code accompanying the GradAttn paper. GradAttn replaces fixed residual connections in ResNet with attention-controlled gradient pathways, using a transformer encoder over multi-scale CNN features to dynamically route gradients across network hierarchies. For the full problem formulation, architecture derivation, and results analysis, refer to the paper.

---

## Notebooks

Each notebook is self-contained and covers one dataset. Within each notebook, cells are organized sequentially: the gradient flow analysis tooling, the ResNet-18 baseline, and the three GradAttn variants (No PE, Learnable PE, RoPE). The Tiny ImageNet, SVHN and the FashionMNIST notebooks additionally includes SENet and CBAM baselines used in the attention-augmented comparison.

| Notebook | Dataset | Domain |
|---|---|---|
| `GradAttn_Tiny-ImageNet.ipynb` | Tiny ImageNet | Natural images (200 classes) |
| `GradAttn_CIFAR-10.ipynb` | CIFAR-10 | Natural images (10 classes) |
| `GradAttn_SVHN.ipynb` | SVHN | Street-view digit recognition |
| `GradAttn_FashionMNIST.ipynb` | Fashion-MNIST | Fashion item recognition |
| `GradAttn_TissueMNIST.ipynb` | TissueMNIST (MedMNIST) | Kidney tissue pathology |
| `GradAttn_BloodMNIST.ipynb` | BloodMNIST (MedMNIST) | Blood cell classification |
| `GradAttn_PCam.ipynb` | PatchCamelyon (PCam) | Histopathology (metastasis detection) |
| `GradAttn_PAD-UFES-20.ipynb` | PAD-UFES-20 | Skin lesion classification |

---

## Training Configuration

All models use identical preprocessing pipelines. Augmentation is applied only to training splits; validation and test splits receive resizing and normalization only. Dataset-specific augmentation details are documented within each notebook.

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 128 (64 for PAD-UFES-20) |
| Maximum epochs | 100 |
| Random seed | 42 |
| LR scheduler | ReduceLROnPlateau (factor 0.2, patience 3, min 1e-7) |
| Early stopping | Patience 7, monitored on validation accuracy |
| Best weight restoration | Enabled |

Transformer encoder configuration across all GradAttn variants: embedding dimension 256, feedforward dimension 512, 8 attention heads, 3 encoder layers, dropout 0.1.

---

## Gradient Flow Analysis

All notebooks include a `GradientFlowAnalyzer` class that instruments models via PyTorch backward hooks on convolutional, linear, and transformer encoder layers. It collects per-layer gradient norms and statistics over a configurable number of batches, computes a Gradient Health Score (fraction of layers with gradient norms in [1e-6, 10]), flags vanishing and exploding layers, and produces visualization and text reports. The Tiny ImageNet notebook extends this with a sensitivity analysis over multiple threshold pairs.

---

## Requirements

The notebooks are intended for Google Colab with GPU acceleration. Install the following before running locally:

```
pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm
```

Dataset-specific packages:

| Package | Required for |
|---|---|
| `medmnist` | BloodMNIST, TissueMNIST |
| `h5py` | PatchCamelyon (PCam) |
| `scipy` | SVHN |
| `Pillow` | SVHN, PatchCamelyon (PCam), Tiny ImageNet, PAD-UFES-20 |
| `pandas` | PAD-UFES-20 |
| `opencv-python` | Tiny ImageNet |

---

## Usage

Open any notebook directly in Google Colab via the badge at the top of the file. Dataset downloading and preprocessing are handled automatically within each notebook.

To run locally:

```
git clone https://github.com/SoudeepGhoshal/GradAttn.git
cd GradAttn
jupyter notebook
```

---

## Citation

```
A citation entry will be added upon publication.
```

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.
