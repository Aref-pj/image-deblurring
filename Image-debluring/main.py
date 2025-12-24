import numpy as np
import cv2
from utils.image_io import load_grayscale, show, save_image
from blur.gaussian import gaussian_kernel
from deblur.deconvolution import deblur
from metrices.sharpness import quality

# data path
input_path = "data/blurred.png"
output_path = "data/deblurred_result.png"

# load image in grayscale and calculate the quality
image = load_grayscale(input_path)
initial_quality = quality(image)

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
    
    # Model: Ax = b (A: Blur Operator, x: Latent Image, b: Observed Blur)
    # Solving for x: x = A^-1 * b 
    # In Frequency Domain: x_fft = b_fft / (A_fft + eps)
    # eps suppresses noise amplification (Regularization).
    restored = deblur(image, kernel, eps=0.001) 
    
    # Pixel Intensity Constraints: Map restored values back to [0, 255].
    temp_img = np.clip(restored, 0, 255).astype(np.uint8)
    
    # Edge-Preserving Denoising: Remove "Ringing Artifacts" without blurring edges.
    temp_img = cv2.medianBlur(temp_img, 3)
    
    # calculate quality
    current_score = quality(temp_img)
    print(f"Sigma: {sigma:.1f} | Score: {current_score:.4f}")

    if current_score > best_score:
        best_score = current_score
        best_sigma = sigma
        best_image = restored

# final process
final_image = np.clip(best_image, 0, 255).astype(np.uint8)
final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)
final_image = cv2.medianBlur(final_image, 3)

######
final_quality = quality(final_image)
improvement = ((final_quality - initial_quality) / initial_quality) * 100

print("\n" + "="*40)
print("           FINAL COMPARISON REPORT")
print("="*40)
print(f"Input Image Quality:   {initial_quality:.4f}")
print(f"Output Image Quality:  {final_quality:.4f}")
print(f"Total Improvement:     {improvement:.2f}%")
print(f"Optimal Sigma Found:   {best_sigma:.1f}")
print("="*40)

save_image(output_path, final_image)
show(f"Result (Sigma {best_sigma:.1f})", final_image)