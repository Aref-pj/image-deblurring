# 🖼️ Automated Image Deblurring using Wiener Deconvolution

This project implements an **automated image restoration system** that recovers blurred images using **Wiener Deconvolution** in the frequency domain.  
The system intelligently estimates the optimal blur parameter by iterating over a range of Gaussian blur values and selecting the one that maximizes image sharpness while suppressing noise.

**New Feature:** The system now supports **both grayscale and color images** by independently processing each RGB channel.

This project is suitable for coursework in **Linear Algebra**, **Signal Processing**, or **Image Processing**, and is designed with modular, extensible code architecture.

---

## 📂 Project Structure

```text
├── blur/           # Tools for generating synthetic Gaussian blur (PSF modeling)
├── deblur/         # Core computational logic (FFT + Wiener Filter)
├── metrics/        # Image quality assessment (L1 & L2 norms of gradients)
├── data/           # Input blurred images and output restored results
└── main.py         # Entry point: controls the optimization loop (includes I/O operations)
```

---

## 🧠 Technical Overview

### 1. Gaussian Blur Modeling (Point Spread Function)

Image blur is modeled using a **spatially invariant Gaussian Point Spread Function (PSF)**:

$$
G(x,y) = \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
$$

Where:
- \( \sigma \) controls the blur intensity
- The kernel is **centered at the origin** to preserve symmetry
- The kernel is **normalized** so that its sum equals 1, ensuring brightness preservation

---

### 2. Frequency Domain Processing (FFT)

Direct spatial convolution with a \(K \times K\) kernel has complexity:

$$
O(N^2 K^2)
$$

FFT-based convolution reduces this to:

$$
O(N^2 \log N)
$$

---

### 3. Wiener Deconvolution Filter

$$
\hat{F}(u,v) =
\frac{H^*(u,v)}{|H(u,v)|^2 + \epsilon} \cdot G(u,v)
$$

Where:
- \( H^*(u,v) \) is the complex conjugate of the PSF
- \( \epsilon \) is the regularization parameter

---

### 4. Automated Blur Parameter Optimization

The algorithm iterates over candidate \( \sigma \) values and selects the one maximizing sharpness based on **L1/L2 norms of image gradients**.

---

### 5. Color Image Processing

For color images, the algorithm:

- Processes each RGB channel independently using the same Wiener filter
- Applies median blur for ringing artifact removal on each channel separately
- Uses grayscale conversion only for quality assessment (focusing on sharpness, not color)
- Saves the final output as a full-color image

---


## 🚀 Usage

```bash
pip install -r requirements.txt
python main.py
```

---

## 🛠️ Key Features

- FFT-based high performance
- Robust Wiener regularization
- Fully automated blur estimation

---

## ⚠️ Limitations

- Assumes Gaussian blur
- Sensitive to strong non-Gaussian noise

---