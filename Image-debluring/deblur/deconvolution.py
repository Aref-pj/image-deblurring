import numpy as np
import cv2

def deblur(image, kernel, eps=0.001):
    """
    Performs Wiener Deconvolution to restore a blurred image.
    This method balances blur removal and noise suppression.
    """
    
    # 1. Pre-processing: Edge Tapering / Smoothing
    # Combining a slightly blurred version with the original helps to reduce 
    # 'Ringing Artifacts' at sharp edges during the frequency domain operations.
    image = cv2.GaussianBlur(image, (5, 5), 0) * 0.05 + image * 0.95
    
    # 2. Transform Image to Frequency Domain
    # Use 2D Fast Fourier Transform (FFT) to diagonalize the convolution operator.
    F_image = np.fft.fft2(image)
    
    # 3. Transform Kernel to Frequency Domain
    # The kernel is zero-padded to match the image size (s=image.shape).
    F_kernel = np.fft.fft2(kernel, s=image.shape)
    
    # 4. Wiener Filter Construction (The core math)
    # F_kernel_conj: Complex conjugate of the kernel to correct the Phase.
    F_kernel_conj = np.conj(F_kernel)
    
    # denom: Power Spectral Density of the kernel + Regularization term (eps).
    # eps acts as a safeguard to prevent division by zero and limit noise amplification.
    denom = np.abs(F_kernel)**2 + eps
    
    # 5. Frequency Domain Division (Restoration)
    # Instead of direct division (F_image / F_kernel), we multiply by the Wiener inverse.
    # This operation is element-wise due to the diagonalization in the Fourier space.
    F_result = (F_kernel_conj / denom) * F_image
    
    # 6. Transform back to Spatial Domain
    # Inverse FFT converts the restored frequency components back to pixel intensities.
    result = np.fft.ifft2(F_result)
    
    # 7. Final Output Cleaning
    # Return only the real part (discarding tiny imaginary errors from computation).
    return np.real(result)