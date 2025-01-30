import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Paths to NIH dataset (change these paths accordingly)
high_res_dir = 'NIH_ChestXray_HR'  # Path to high-res chest X-ray images
low_res_dir = 'NIH_ChestXray_LR'  # Path to store low-res versions

# Image dimensions 
high_res_size = (256, 256)  # Target high-resolution size
low_res_size = (64, 64)     # Target low-resolution size

# Define the image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize(high_res_size),  # Resized to HR size
    transforms.Grayscale(num_output_channels=1),  # Converted to grayscale (1 channel)
    transforms.ToTensor(),  # Converted to tensor (scales to [0, 1])
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalized to [-1, 1] for single channel
])

# Custom Dataset class to handle NIH Chest X-ray dataset loading
class NIHChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, high_res_dir, low_res_size, high_res_size, transform=None):
        self.high_res_dir = high_res_dir
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(high_res_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load high-res image
        high_res_image_path = os.path.join(self.high_res_dir, self.image_files[idx])
        hr_img = Image.open(high_res_image_path).convert('L')  # Converted to grayscale

        # Downscale to low-resolution
        lr_img = hr_img.resize(self.low_res_size, Image.BICUBIC)

        # Apply transformation (if any)
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = transforms.Compose([
                transforms.Resize(self.low_res_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])(lr_img)

        return lr_img, hr_img

# Create the dataset instance
dataset = NIHChestXrayDataset(high_res_dir, low_res_size, high_res_size, transform=transform)

# Process each image in the dataset
for i in range(len(dataset)):
    lr_img, hr_img = dataset[i]  

    # Save the low-res images for future use
    save_img_path = os.path.join(low_res_dir, f"low_res_image_{i+1}.png")
    lr_img_numpy = lr_img.squeeze(0).cpu().numpy() 
    lr_img_numpy = (lr_img_numpy * 127.5 + 127.5).astype(np.uint8)  
    Image.fromarray(lr_img_numpy).save(save_img_path)

    # Print progress
    if i % 100 == 0:
        print(f"Processed {i+1}/{len(dataset)} images...")

print("Preprocessing complete!")
