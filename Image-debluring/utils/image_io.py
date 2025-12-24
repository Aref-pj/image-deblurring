import cv2
import numpy as np

def load_grayscale(path):
    """
    Loads an image from the disk in grayscale mode and converts it to float.
    Using float is essential for high-precision mathematical operations like FFT.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at {path}")
    
    # Convert uint8 [0, 255] to float for precise linear algebra calculations
    return img.astype(float)

def show(title, img):
    """
    Displays the image in a window. 
    Handles data conversion back to uint8 before rendering.
    """
    # Clip values to [0, 255] range to prevent visualization artifacts (Overflow)
    display_img = np.clip(img, 0, 255).astype("uint8")
    cv2.imshow(title, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(path, img):
    """
    Saves the processed image to the specified path.
    Ensures the data is in the standard 8-bit integer format.
    """
    # Ensure all pixel values are within the valid 0-255 range
    img = np.clip(img, 0, 255)
    # Cast back to uint8 for standard image file compatibility (PNG/JPG)
    cv2.imwrite(path, img.astype("uint8"))