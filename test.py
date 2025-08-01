import os
import cv2
import numpy as np
import random
from detector import advanced_blur_detect
from utils import display_intermediate_results, find_images


def generate_out_of_focus(image: np.ndarray, level: int) -> np.ndarray:
    """
    Applies an out-of-focus blur to an image using OpenCV.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        level (int): The desired level of blur, from 1 (slight) to 5 (heavy).

    Returns:
        np.ndarray: The blurred image as a NumPy array.
    """
    # Clamp the level to be within the 1-5 range
    level = max(1, min(level, 5))

    # Map the level (1-5) to an odd kernel size.
    # A larger kernel size results in a more significant blur.
    # We multiply by 2 and add 1 to ensure the kernel size is always odd.
    # Example mapping: Level 1 -> 9x9, Level 5 -> 41x41
    blur_intensity = (level * 4 * 2) + 1
    kernel_size = (blur_intensity, blur_intensity)

    # Apply Gaussian blur
    # The third argument (sigmaX) is set to 0, which tells OpenCV
    # to calculate it automatically based on the kernel size.
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred_image


def test_blur_detection_on_random_blur(input_dir):
    """
    For each image in input_dir, apply a random out-of-focus blur (or leave sharp),
    then detect blur and show the result. Continue by key press.
    """
    image_paths = list(find_images(input_dir))
    print(f"Found {len(image_paths)} images in {input_dir}")
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        # Randomly decide blur level: 0 (sharp) or 1-5 (blur)
        blur_level = random.choice([1, 2, 3, 4, 5])
        if blur_level > 0:
            img_blur = generate_out_of_focus(img, blur_level)
            label = f"Blurred (level {blur_level})"
        else:
            img_blur = img.copy()
            label = "Sharp (no blur)"
        # Detect blur
        result = advanced_blur_detect(img_blur, threshold=35, min_zero_threshold=0.001, conservative_threshold=0.7, feature_sensitivity=0.75, dark_threshold=50, enable_dark_enhancement=False)
        # Show result and wait for key
        print(f"{os.path.basename(img_path)}: {label} | Detected: {'Blur' if result['classification'] else 'Sharp'} | Quality: {result['quality_score']} | BlurExtent: {result['blur_extent']:.3f} | Per: {result['per']:.5f}")
        key = display_intermediate_results(img_blur, result, img_path)
        if key == ord('q'):
            print("Exiting test...")
            break

# Example usage:
if __name__ == "__main__":
    test_input_dir = "E:\Dataset\OR"  # Change to your test folder
    test_blur_detection_on_random_blur(test_input_dir)