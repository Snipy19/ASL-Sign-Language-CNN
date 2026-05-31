# ASL Sign Language Recognition — CNN (PyTorch)

A Convolutional Neural Network trained to classify 24 American Sign Language (ASL) hand gestures from grayscale images, achieving **98.69% validation accuracy**.

---

## Overview

This project implements an image classification pipeline for ASL gesture recognition using the Sign MNIST dataset. The model is trained entirely from scratch in PyTorch with GPU acceleration via CUDA.

The goal was not just to achieve high accuracy, but to build a clean, well-structured pipeline — from raw data loading to a trained model — with proper regularization and a disciplined training loop.

---

## Dataset

- **Source:** Sign MNIST (Kaggle)
- **Classes:** 24 ASL alphabet gestures (J and Z excluded as they require motion)
- **Image size:** 28 × 28 grayscale
- **Train samples:** 27,455
- **Validation samples:** 7,172

---

## Model Architecture

Three convolutional blocks followed by fully connected layers:

```
Conv2d(1, 25, 3)  → BatchNorm → ReLU → MaxPool(2)
Conv2d(25, 50, 3) → BatchNorm → ReLU → Dropout(0.2) → MaxPool(2)
Conv2d(50, 75, 3) → BatchNorm → ReLU → MaxPool(2)
Flatten
Linear(675, 512)  → Dropout(0.3) → ReLU
Linear(512, 24)
```

---

## Results

| Epoch | Train Accuracy | Validation Accuracy |
|-------|---------------|---------------------|
| 1     | 91.19%        | 96.85%              |
| 10    | 99.76%        | 97.21%              |
| 18    | 100.00%       | 98.69%              |
| 20    | 99.95%        | 98.40%              |

**Best validation accuracy: 98.69% (Epoch 18)**

---

## Training Details

| Parameter       | Value              |
|-----------------|--------------------|
| Optimizer       | Adam (default lr)  |
| Loss Function   | CrossEntropyLoss   |
| Batch Size      | 32                 |
| Epochs          | 20                 |
| Device          | CUDA (GPU)         |
| Normalization   | Pixel values / 255 |

---

## Project Structure

```
├── notebook.ipynb       # Full training pipeline
├── data/
│   └── asl_data/
│       ├── sign_mnist_train.csv
│       └── sign_mnist_valid.csv
└── README.md
```

---

## Requirements

```
torch
torchvision
pandas
numpy
```

Install with:

```bash
pip install torch torchvision pandas numpy
```

---

## How to Run

1. Download the Sign MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and place CSVs in `data/asl_data/`
2. Open `notebook.ipynb` in JupyterLab
3. Run all cells

---

## Key Learnings

- BatchNorm between convolutional layers significantly stabilized training
- Dropout placement (after second and before final dense layer) improved generalization without hurting training speed
- `torch.compile()` with the Inductor backend requires a working Triton installation — handled gracefully with a try/except fallback

---

## Author

**Dhruv Raj Ghai**  
Electronics & AI/ML Enthusiast | 3rd Year ECE, Thapar University  
[LinkedIn](https://www.linkedin.com/in/your-linkedin) · [GitHub](https://github.com/Snipy19)
