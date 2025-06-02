# 🎨 Closed-Form Matting Project

This project provides a robust Python-based implementation of **Closed-Form Image Matting**, a technique used to accurately extract foreground objects from images based on user-provided scribbles. The method leverages advanced linear algebra (Laplacian matting formulation) to generate precise alpha mattes indicating pixel-level transparency.

## 📚 Overview of the Methodology

Closed-Form Matting is based on solving the following key computational steps:

1. **Matting Laplacian Construction**  
   Constructs a sparse matrix encoding pixel affinities based on local color distributions within image neighborhoods.

2. **Alpha Matte Computation**  
   Solves a large, sparse linear system derived from user scribbles and image data, resulting in a continuous alpha channel indicating foreground-background separation.

3. **Automatic Image and Scribble Alignment**  
   The provided images and scribbles are automatically oriented, resized, and aligned to ensure compatibility and optimal memory usage during computation.

---

## 📁 Project Structure

```
IACVProject/
├── assets/              # Original images and scribbles for matting
│   ├── images/
│   └── scribbles/
├── output/              # Resulting alpha mattes
│   └── alpha/
├── notebooks/           # Analysis and visualization notebooks
├── papers/              # Reference papers and resources
└── scripts/             # Core implementation scripts and modules
    ├── compute_alpha.py
    └── src/
        ├── closed_form_matting.py               # Core Laplacian computations
        ├── solve_foreground_background.py       # Foreground extraction utilities
        └── __init__.py
```

---

## 🛠️ Computational Workflow

The main computational process is implemented as follows:

- **Image Pre-processing**
  - Orientation and dimensional consistency checks.
  - Safe resizing of images exceeding memory thresholds.
  - Automatic color-channel adjustments for scribbles.

- **Matting Laplacian Matrix Computation**
  - Calculation of a sparse affinity matrix that encodes pixel similarity.

- **Linear System Solution**
  - Solving for alpha transparency using sparse linear algebra techniques.

- **Post-processing**
  - Generation of an alpha matte with pixel values between `[0, 1]`.
  - Optional resizing back to original image dimensions.

---

## 🧬 Dependencies & Environment

The core computational scripts rely on:

- NumPy: Array manipulations and linear algebra computations.
- OpenCV: Image loading, resizing, and saving results efficiently.

Ensure dependencies are installed using:

```bash
pip install numpy opencv-python
```

---

## 📓 Analysis & Visualization

- Jupyter notebooks are provided for exploring matting results, analyzing accuracy, and visualizing the alpha mattes generated.

---

## 🔖 Reference & Theory

The methodology implemented in this project is derived from the seminal work:

- **"A Closed-Form Solution to Natural Image Matting"**, Anat Levin, Dani Lischinski, and Yair Weiss (2008).

---

## ✍️ Contributors

- **Davide Franchi**
- **Biaggio Cancelliere**
- **Carlos Ruiz**
