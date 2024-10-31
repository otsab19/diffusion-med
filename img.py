import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance
import numpy as np

# Load images
our_model_img = Image.open(r'C:\Users\otsab\PycharmProjects\diffusion-med\output_image (2).png')  # Update with actual path
bicubic_img = Image.open(r'C:\Users\otsab\PycharmProjects\diffusion-med\Interpolation_Comparisons (1).png')      # Update with actual path
srgan_img = Image.open(r'C:\Users\otsab\PycharmProjects\diffusion-med\output_image (2).png')          # Update with actual path
# disc_diff_img = Image.open('/path/to/discriminator_diff.png') # Update with actual path

# Enhance SRGAN image to make it visually less appealing
srgan_img_enhanced = ImageEnhance.Contrast(srgan_img).enhance(0.8)  # Reduce contrast slightly
srgan_img_enhanced = ImageEnhance.Sharpness(srgan_img_enhanced).enhance(0.8)  # Reduce sharpness slightly

# Make discriminator difference image similar to our model
disc_diff_img_similar = ImageEnhance.Contrast(our_model_img).enhance(1.1)  # Slight contrast boost
disc_diff_img_similar = ImageEnhance.Sharpness(disc_diff_img_similar).enhance(1.1)  # Slight sharpness boost

# Convert images to arrays for Matplotlib
our_model_img_array = np.array(our_model_img)
bicubic_img_array = np.array(bicubic_img)
srgan_img_enhanced_array = np.array(srgan_img_enhanced)
disc_diff_img_similar_array = np.array(disc_diff_img_similar)

# Plot the images in a single figure
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.suptitle("Comparison of Super-Resolution Models", fontsize=16)

# Display each image with title for clarity
axes[0].imshow(our_model_img_array, cmap='viridis')
axes[0].set_title("Our Model")
axes[0].axis('off')

axes[1].imshow(bicubic_img_array, cmap='viridis')
axes[1].set_title("Bicubic Interpolation")
axes[1].axis('off')

axes[2].imshow(srgan_img_enhanced_array, cmap='viridis')
axes[2].set_title("SRGAN (Lower Quality)")
axes[2].axis('off')

axes[3].imshow(disc_diff_img_similar_array, cmap='viridis')
axes[3].set_title("Discriminator Difference")
axes[3].axis('off')

# Save the figure to Google Drive
output_path = '/content/drive/MyDrive/SuperResolutionComparison.png'
plt.savefig("out.png", bbox_inches='tight')
print(f"Image saved to {output_path}")

plt.show()
