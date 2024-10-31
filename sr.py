from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageEnhance


def fake_srgan_style_right(image_np, sharpness_factor=1.5, contrast_factor=1.2, blur_strength=5):
    """
    Apply an SRGAN-like style effect to the right half of the image, with slight anatomical degradation.
    :param image_np: Input image as a numpy array (range [0, 1]).
    :param sharpness_factor: Factor to increase sharpness on the right side.
    :param contrast_factor: Factor to increase contrast on the right side.
    :param blur_strength: Strength of blurring to reduce fine anatomical details.
    :return: Modified image with SRGAN-like effect applied to the right half only.
    """
    height, width, _ = image_np.shape
    mid = width // 2

    # Split the image into left and right halves
    left_half = image_np[:, :mid]
    right_half = image_np[:, mid:]

    # Convert right half to PIL Image for contrast and sharpness adjustment
    right_half_pil = Image.fromarray((right_half * 255).astype(np.uint8))

    # Step 1: Enhance sharpness and contrast
    sharpness_enhancer = ImageEnhance.Sharpness(right_half_pil)
    right_half_pil = sharpness_enhancer.enhance(sharpness_factor)

    contrast_enhancer = ImageEnhance.Contrast(right_half_pil)
    right_half_pil = contrast_enhancer.enhance(contrast_factor)

    # Convert back to numpy array
    right_half_np = np.array(right_half_pil) / 255.0

    # Step 2: Apply blur to mimic loss of fine anatomical details
    right_half_np = cv2.GaussianBlur(right_half_np, (blur_strength, blur_strength), 0)

    # Combine the left half and modified right half
    modified_image = np.hstack((left_half, right_half_np))
    return modified_image


# Load the PNG image
input_image_path = "grayscale_image.jpg"
image = Image.open(input_image_path).convert("RGB")
image_np = np.array(image) / 255.0  # Normalize to [0, 1]

# Apply SRGAN-like styling
fake_srgan_image_np = fake_srgan_style_right(image_np)

# Convert back to image format and save or display
fake_srgan_image = Image.fromarray((fake_srgan_image_np * 255).astype(np.uint8))
fake_srgan_image.save("fake_srgan_output.png")
fake_srgan_image.show()
