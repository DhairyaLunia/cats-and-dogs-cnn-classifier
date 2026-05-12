# 🐶🐱 Dogs vs. Cats — EfficientNet-B3 Image Classifier

> Kaggle Competition: [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview)  
> Task: Binary image classification (Dog = 1, Cat = 0)  
> Evaluation Metric: Log Loss

---

## Overview

This project fine-tunes a pretrained **EfficientNet-B3** model on the Kaggle Dogs vs. Cats dataset using PyTorch. It incorporates several advanced training techniques — progressive unfreezing, Mixup augmentation, gradient accumulation, and mixed-precision training — to achieve strong generalization from a compact architecture.

---

## Techniques Used

| Technique | Detail |
|---|---|
| **Model** | EfficientNet-B3 (ImageNet pretrained) |
| **Transfer Learning** | Frozen backbone; only classifier + last 4 feature blocks trained initially |
| **Progressive Unfreezing** | Full backbone unfrozen at epoch 3 |
| **Mixup Augmentation** | α = 0.4, interpolates image pairs and labels |
| **Gradient Accumulation** | Effective batch size = 24 × 6 = **144** |
| **Mixed Precision (AMP)** | `torch.amp.autocast` + `GradScaler` for faster GPU training |
| **LR Scheduling** | Linear warmup (1 epoch) → Cosine Annealing |
| **Optimizer** | AdamW with differential learning rates per layer group |
| **Early Stopping** | Patience = 3 epochs on validation loss |
| **Loss** | BCEWithLogitsLoss |

---

## Data Augmentation

**Training:**
- RandomResizedCrop (scale 0.65–1.0)
- RandomHorizontalFlip / RandomVerticalFlip (p=0.1)
- RandomRotation (±25°)
- ColorJitter (brightness, contrast, saturation, hue)
- RandomGrayscale (p=0.05)
- RandomErasing (p=0.25)
- ImageNet normalization

**Validation / Test:**
- Resize to 380×380
- ImageNet normalization

---

## Model Architecture

```
EfficientNet-B3 (pretrained backbone)
    └── classifier
        ├── BatchNorm1d(1536)
        ├── Dropout(0.3)
        └── Linear(1536 → 1)   # binary output (logit)
```

Weight init: Xavier uniform (weights), zeros (bias) on the final linear layer.

---

## Training Setup

```
Image size    : 380 × 380
Batch size    : 24 (effective: 144 via gradient accumulation)
Epochs        : 5 (with early stopping, patience=3)
Optimizer     : AdamW
  - Classifier LR     : 3e-4 | weight_decay: 1e-4
  - features[-4:] LR  : 5e-5 | weight_decay: 1e-4
  - features[:-4] LR  : 1e-5 | weight_decay: 1e-4  (after epoch 3)
LR Schedule   : 1-epoch warmup → CosineAnnealingLR (eta_min=1e-6)
Device        : CUDA (GPU)
```

---

## Project Structure

```
├── predictive-analytics-q2.ipynb   # Main notebook (end-to-end pipeline)
├── final_submission.csv            # Kaggle submission file
└── best_model.pth                  # Saved model checkpoint (best val loss)
```

---

## Pipeline

1. **Unzip** `train.zip` and `test.zip` from Kaggle input
2. **Parse labels** from filenames (`dog.*` → 1, `cat.*` → 0)
3. **80/20 stratified split** for train/validation
4. **Build EfficientNet-B3** with custom classifier head
5. **Train** with Mixup, AMP, gradient accumulation, progressive unfreezing
6. **Evaluate** on validation set each epoch; save best checkpoint
7. **Inference** on test set → sigmoid probabilities
8. **Export** `final_submission.csv` with `id` and `label` columns

---

## Requirements

```bash
pip install torch torchvision pillow numpy pandas scikit-learn
```

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) and place the zips in `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/`
2. Open `predictive-analytics-q2.ipynb` in Kaggle or a local Jupyter environment with a GPU
3. Run all cells — training, validation, and submission generation are fully sequential

---

## References

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
