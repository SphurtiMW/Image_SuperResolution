import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from srgan_model import Generator
import torch
torch.set_num_threads(1)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Paths to images and model
lr_image_path = 'NIH_ChestXray_LR\low_res_image_163.png'
hr_image_path = 'NIH_ChestXray_HR\00030606_013.png'
model_path = 'SRGAN_medical_imaging\fine_tuned_SRGAN.pt'

# Load the SRGAN model
device = torch.device("cpu")
model = Generator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transform to prepare image for SRGAN
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalized to [-1, 1]
])

# Denormalization function
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Scale to [0, 1]
    tensor = tensor * 255.0  # Scale to [0, 255]
    return tensor.clamp(0, 255).byte().cpu().numpy()

# Load low-res and high-res images
lr_img = Image.open(lr_image_path).convert('L')  
hr_img = Image.open(hr_image_path).convert('L')  

# Resize low-res image for SRGAN model input
lr_img = lr_img.resize((64, 64), Image.BICUBIC)

# Apply transformations to the images
lr_img_tensor = transform(lr_img).unsqueeze(0).to(device)  
hr_img_tensor = transform(hr_img).unsqueeze(0).to(device)

# Generate super-resolved image
with torch.no_grad():
    sr_output = model(lr_img_tensor)
    
    # Handle tuple output from the model
    if isinstance(sr_output, tuple):
        sr_img_tensor = sr_output[0]  
    else:
        sr_img_tensor = sr_output

# Convert tensors to numpy arrays for visualization
lr_img_np = lr_img.resize((256, 256), Image.BICUBIC)  
sr_img_np = denormalize(sr_img_tensor.squeeze(0).squeeze(0))
hr_img_np = denormalize(hr_img_tensor.squeeze(0).squeeze(0))

# Plot the images side by side
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(lr_img_np, cmap='gray')
plt.title('Low-Resolution (Input)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sr_img_np, cmap='gray')
plt.title('Super-Resolved (Output)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(hr_img_np, cmap='gray')
plt.title('High-Resolution (Ground Truth)')
plt.axis('off')

plt.tight_layout()
plt.show()

output_dir = r'C:\Users\sphur\Downloads\SRGAN_medical_imaging\outputs'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the images
lr_img.save(os.path.join(output_dir, 'low_res_image.png'))
Image.fromarray(sr_img_np).save(os.path.join(output_dir, 'super_res_image.png'))
Image.fromarray(hr_img_np).save(os.path.join(output_dir, 'high_res_image.png'))

print(f"Images saved in {output_dir}")

