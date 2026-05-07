# main.py
import numpy as np
import cv2
from blur.gaussian import gaussian_kernel
from deblur.deconvolution import deblur
from metrices.sharpness import quality

# data path
input_path = "data/blurred1.png"
output_path = "data/deblurred_result.png"

# Modified: Load image in color and calculate the quality
image = cv2.imread(input_path)
if image is None:
    raise ValueError(f"Image not found at {input_path}")
image = image.astype(float)

# Modified: Calculate quality on grayscale version for initial quality
initial_quality = quality(cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY))

# initial var
best_sigma = None
best_score = -1e9
best_image = None

print(f"Initial Quality: {initial_quality:.4f}")
print("Searching for optimal Sigma...")

# sigma range -> 0.1 - 4.0 , steps -> 0.1
sigmas = np.arange(0.1, 4, 0.1)

for sigma in sigmas:
    # Generate Gaussian Kernel (Point Spread Function)
    kernel = gaussian_kernel(sigma)
    
    # Modified: Process each channel separately for color image
    if len(image.shape) == 3:  # Color image (H, W, C)
        restored_channels = []
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            # Model: Ax = b (A: Blur Operator, x: Latent Image, b: Observed Blur)
            # Solving for x: x = A^-1 * b 
            # In Frequency Domain: x_fft = b_fft / (A_fft + eps)
            # eps suppresses noise amplification (Regularization).
            restored_channel = deblur(channel, kernel, eps=0.001)
            restored_channels.append(restored_channel)
        restored = np.stack(restored_channels, axis=2)
    else:  # Grayscale image
        # Model: Ax = b (A: Blur Operator, x: Latent Image, b: Observed Blur)
        # Solving for x: x = A^-1 * b 
        # In Frequency Domain: x_fft = b_fft / (A_fft + eps)
        # eps suppresses noise amplification (Regularization).
        restored = deblur(image, kernel, eps=0.001)
    
    # Pixel Intensity Constraints: Map restored values back to [0, 255].
    temp_img = np.clip(restored, 0, 255).astype(np.uint8)
    
    # Modified: Apply median blur to each channel separately for color
    if len(temp_img.shape) == 3:
        # Edge-Preserving Denoising: Remove "Ringing Artifacts" without blurring edges.
        temp_img_channels = []
        for c in range(temp_img.shape[2]):
            temp_img_channels.append(cv2.medianBlur(temp_img[:, :, c], 3))
        temp_img = np.stack(temp_img_channels, axis=2)
    else:
        # Edge-Preserving Denoising: Remove "Ringing Artifacts" without blurring edges.
        temp_img = cv2.medianBlur(temp_img, 3)
    
    # Modified: Calculate quality on grayscale version
    if len(temp_img.shape) == 3:
        gray_temp = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_temp = temp_img
    current_score = quality(gray_temp)
    print(f"Sigma: {sigma:.1f} | Score: {current_score:.4f}")

    if current_score > best_score:
        best_score = current_score
        best_sigma = sigma
        best_image = restored

# final process
final_image = np.clip(best_image, 0, 255).astype(np.uint8)

# Modified: Normalize each channel separately for color image
if len(final_image.shape) == 3:
    final_image_channels = []
    for c in range(final_image.shape[2]):
        final_image_channels.append(cv2.normalize(final_image[:, :, c], None, 0, 255, cv2.NORM_MINMAX))
    final_image = np.stack(final_image_channels, axis=2)
else:
    final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)

# Modified: Apply median blur to each channel separately for color image
if len(final_image.shape) == 3:
    final_image_channels = []
    for c in range(final_image.shape[2]):
        final_image_channels.append(cv2.medianBlur(final_image[:, :, c], 3))
    final_image = np.stack(final_image_channels, axis=2)
else:
    final_image = cv2.medianBlur(final_image, 3)

######
# Modified: Calculate final quality on grayscale version
if len(final_image.shape) == 3:
    final_gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
else:
    final_gray = final_image
final_quality = quality(final_gray)
improvement = ((final_quality - initial_quality) / initial_quality) * 100

print("\n" + "="*40)
print("           FINAL COMPARISON REPORT")
print("="*40)
print(f"Input Image Quality:   {initial_quality:.4f}")
print(f"Output Image Quality:  {final_quality:.4f}")
print(f"Total Improvement:     {improvement:.2f}%")
print(f"Optimal Sigma Found:   {best_sigma:.1f}")
print("="*40)

# Modified: Save image directly without using image_io
cv2.imwrite(output_path, final_image)

# Modified: Removed show() as requested - no image display