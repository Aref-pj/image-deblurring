import cv2
import numpy as np

def quality(image):
    """
    Computes a sharpness score based on the Sparsity of the image gradients.
    It measures how 'concentrated' the edge energy is, which correlates with clarity.
    """
    
    # 1. Ensure the image is in 8-bit format (Standard for OpenCV operators)
    if image.dtype != np.uint8:
        # Clip values to [0, 255] to prevent overflow and cast to uint8
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image

    # 2. Edge Detection using Sobel Operators (Approximating the Image Derivatives)
    # Calculate horizontal (x) and vertical (y) gradients separately
    sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Compute Gradient Magnitude Matrix
    # This matrix holds the 'edge strength' for every single pixel
    # magnitude[i,j] = sqrt(sobelx[i,j]^2 + sobely[i,j]^2)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 4. Dimensionality Reduction: Convert the matrix into scalar statistical norms
    # L1 Norm: The simple sum of all edge intensities (Total energy)
    l1_norm = np.sum(magnitude)

    # L2 Norm: The Euclidean norm (Energy concentrated in sharp peaks)
    # L2 gives much more weight to high-value pixels (sharp edges)
    l2_norm = np.sqrt(np.sum(magnitude**2))

    # 5. Handle the case of a perfectly flat/uniform image to avoid division by zero
    if l1_norm == 0: 
        return 0
    
    # 6. Final Sharpness Metric (Ratio of L2 over L1)
    # Mathematically, this ratio increases as the image becomes 'sharper' and less 'blurred'.
    # 1e-6 is a safety epsilon to ensure numerical stability.
    score = l2_norm / (l1_norm + 1e-6)
    
    return score