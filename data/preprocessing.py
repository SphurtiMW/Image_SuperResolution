import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Paths to DIV2K dataset
high_res_dir = ''  # Path to high-res DIV2K images
low_res_dir = ''   # Path  to low-res images

# Image dimensions (HR size, typically 4x the LR size for SRGANs)
high_res_size = (256, 256)
low_res_size = (64, 64)

# Load and preprocess images
def preprocess_image(image_path, low_res_size, high_res_size):
    # Load high-res image
    img = load_img(image_path, target_size=high_res_size)
    hr_img = img_to_array(img)

    # Downscale to low-resolution (simulating LR input)
    lr_img = img.resize(low_res_size, Image.BICUBIC)
    lr_img = img_to_array(lr_img)

    # Normalize images to [-1, 1] range for the SRGAN model
    hr_img = (hr_img / 127.5) - 1
    lr_img = (lr_img / 127.5) - 1

    return lr_img, hr_img

def load_div2k_images(high_res_dir, low_res_size, high_res_size):
    lr_images, hr_images = [], []
    
    for image_file in os.listdir(high_res_dir):
        image_path = os.path.join(high_res_dir, image_file)
        
        lr_img, hr_img = preprocess_image(image_path, low_res_size, high_res_size)
        lr_images.append(lr_img)
        hr_images.append(hr_img)
    
    return np.array(lr_images), np.array(hr_images)

# Load DIV2K dataset
lr_images, hr_images = load_div2k_images(high_res_dir, low_res_size, high_res_size)

# Verify the shape of the preprocessed data
print(f"Low-resolution images shape: {lr_images.shape}")
print(f"High-resolution images shape: {hr_images.shape}")

# Example output shapes
# Low-resolution images shape: (n_samples, 64, 64, 3)
# High-resolution images shape: (n_samples, 256, 256, 3)

# Save the low-resolution images (optional)
for i, lr_img in enumerate(lr_images):
    save_img_path = os.path.join(low_res_dir, f"low_res_image_{i+1}.png")
    lr_img = ((lr_img + 1) * 127.5).astype(np.uint8)  # Convert back to [0, 255] range
    Image.fromarray(lr_img).save(save_img_path)

