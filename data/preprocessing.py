import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Paths to DIV2K dataset
high_res_dir = r'C:\Users\sphur\Downloads\DIV2K_train_HR'  # Path to high-res DIV2K images
low_res_dir = r'C:\Users\sphur\Downloads\DIV2K_train_LR'   # Path to low-res images

# Image dimensions (HR size, typically 4x the LR size for SRGANs)
high_res_size = (256, 256)
low_res_size = (64, 64)

# Define the image preprocessing transformation for PyTorch
transform = transforms.Compose([
    transforms.Resize(high_res_size),  # Resize image to high-res size
    transforms.ToTensor(),  # Convert to PyTorch tensor and scale to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Custom Dataset class to handle DIV2K dataset loading
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, high_res_dir, low_res_size, high_res_size, transform=None):
        self.high_res_dir = high_res_dir
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.transform = transform
        self.image_files = os.listdir(high_res_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load high-res image
        high_res_image_path = os.path.join(self.high_res_dir, self.image_files[idx])
        hr_img = Image.open(high_res_image_path).convert('RGB')

        # Downscale to low-resolution
        lr_img = hr_img.resize(self.low_res_size, Image.BICUBIC)

        # Apply transformation (if any)
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img

# Create the dataset instance
dataset = DIV2KDataset(high_res_dir, low_res_size, high_res_size, transform=transform)

# Process each image in the dataset
for i in range(len(dataset)):
    lr_img, hr_img = dataset[i]  # Get low-res and high-res images

    # Optionally, save the low-res images
    save_img_path = os.path.join(low_res_dir, f"low_res_image_{i+1}.png")
    lr_img = lr_img.permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC
    lr_img = (lr_img * 127.5 + 127.5).astype(np.uint8)  # Denormalize to [0, 255]
    Image.fromarray(lr_img).save(save_img_path)

    # Print status
    if i % 100 == 0:
        print(f"Processed {i+1}/{len(dataset)} images...")

print("Preprocessing complete!")
