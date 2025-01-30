import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from srgan_model import Generator 

# Set device to CPU explicitly
device = torch.device("cpu")
print("Evaluating on:", device)

# Data transformation for evaluation (ensure images are resized correctly)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Load evaluation dataset
class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, high_res_dir, transform=None):
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(high_res_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.high_res_dir, self.image_files[idx])
        hr_img = Image.open(img_path).convert('L')  
        hr_img = hr_img.resize((256, 256), Image.BICUBIC)
        lr_img = hr_img.resize((64, 64), Image.BICUBIC)

        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img

# Paths to dataset and model
eval_dataset_path = 'NIH_ChestXray_LR'
model_path = 'fine_tuned_SRGAN.pt'

# Load evaluation dataset
eval_dataset = MedicalImageDataset(eval_dataset_path, transform)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
print(f"Number of samples in evaluation dataset: {len(eval_dataset)}")

# Load trained SRGAN model
print("Loading model...")
model = Generator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Fine-tuned model loaded successfully for evaluation.")

# Load VGG model for perceptual loss (use pre-trained weights)
vgg = models.vgg19(pretrained=True).features[:9].to(device).eval()

# Function to compute perceptual loss using VGG19 features
def perceptual_loss(sr_img, hr_img):
    # Convert to 3-channel for VGG input
    sr_img = sr_img.repeat(1, 3, 1, 1)
    hr_img = hr_img.repeat(1, 3, 1, 1)
    sr_features = vgg(sr_img)
    hr_features = vgg(hr_img)
    loss = F.mse_loss(sr_features, hr_features)
    return loss.item()

# Function to denormalize images (convert from [-1,1] to [0,255])
def denormalize(tensor):
    if isinstance(tensor, (list, tuple)):  
        tensor = torch.stack(tensor)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    tensor = tensor * 0.5 + 0.5  # Scaled to [0,1]
    tensor = tensor * 255.0  # Scaled to [0,255]
    return tensor.clamp(0, 255).byte()

# Loss trackers
total_psnr = 0.0
total_ssim = 0.0
total_mse_loss = 0.0
total_l1_loss = 0.0
total_perceptual_loss = 0.0
num_samples = 0

# L1 Loss (Mean Absolute Error)
l1_loss_fn = nn.L1Loss()
mse_loss_fn = nn.MSELoss()

# Evaluate the model on the dataset
with torch.no_grad():
    for lr_imgs, hr_imgs in eval_loader:
        num_samples += 1
        print(f"Processing sample {num_samples}...")

        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        # Generate super-resolved images
        sr_output = model(lr_imgs)
        
        # Handle model output tuple case
        if isinstance(sr_output, tuple):
            sr_imgs = sr_output[0]  
        else:
            sr_imgs = sr_output

        # Compute perceptual loss (VGG features)
        perceptual_loss_value = perceptual_loss(sr_imgs, hr_imgs)

        # Compute L1 Loss
        l1_loss_value = l1_loss_fn(sr_imgs, hr_imgs).item()

        # Compute MSE Loss
        mse_loss_value = mse_loss_fn(sr_imgs, hr_imgs).item()

        # Convert tensors to numpy arrays for metric computation
        sr_imgs_np = denormalize(sr_imgs).cpu().squeeze().numpy()
        hr_imgs_np = denormalize(hr_imgs).cpu().squeeze().numpy()

        # Ensure proper dimensions for SSIM calculation
        if sr_imgs_np.ndim == 3:
            sr_imgs_np = sr_imgs_np[0]  # Removed channel dimension for grayscale
            hr_imgs_np = hr_imgs_np[0]

        # Compute PSNR and SSIM
        psnr_value = psnr(hr_imgs_np, sr_imgs_np, data_range=255)
        ssim_value = ssim(hr_imgs_np, sr_imgs_np, data_range=255)

        total_psnr += psnr_value
        total_ssim += ssim_value
        total_mse_loss += mse_loss_value
        total_l1_loss += l1_loss_value
        total_perceptual_loss += perceptual_loss_value

        print(f"Sample {num_samples} - PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}, MSE: {mse_loss_value:.4f}, L1: {l1_loss_value:.4f}, Perceptual: {perceptual_loss_value:.4f}")

# Compute average metrics
avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples
avg_mse_loss = total_mse_loss / num_samples
avg_l1_loss = total_l1_loss / num_samples
avg_perceptual_loss = total_perceptual_loss / num_samples

print(f"\nEvaluation Results:")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average MSE Loss: {avg_mse_loss:.4f}")
print(f"Average L1 Loss: {avg_l1_loss:.4f}")
print(f"Average Perceptual Loss: {avg_perceptual_loss:.4f}")

print("Evaluation completed!")
