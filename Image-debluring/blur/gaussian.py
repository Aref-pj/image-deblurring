import numpy as np

def gaussian_kernel(sigma):
    """
    Generates a 2D Gaussian kernel (Point Spread Function) to model the blur.
    """
    # 1. Determine kernel size (Rule of thumb: 6 * sigma covers 99.7% of the distribution)
    size = int(6 * sigma)
    if size % 2 == 0: size += 1 # Kernel must have a central pixel (odd size)
    if size < 3: size = 3
    
    # 2. Create a coordinate system centered at (0,0)
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    
    # 3. Apply the 2D Gaussian formula
    # Higher sigma results in a wider, more spread-out "hill" (more blur)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # 4. Normalize the kernel so the sum of all elements equals 1.0
    # This prevents the image from becoming brighter or darker after deconvolution.
    return kernel / np.sum(kernel)