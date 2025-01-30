import torch
from torchvision import transforms
from PIL import Image
import os
from srgan_model import Generator

# Set device to CPU
device = torch.device("cpu")

# Load the SRGAN model
model_path = 'SRGAN_medical_imaging\fine_tuned_SRGAN.pt'
model = Generator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully!")

# Transform for input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalized to [-1, 1]
])

# Denormalization function
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Scaled to [0,1]
    tensor = tensor * 255.0  # Scaled to [0,255]
    return tensor.clamp(0, 255).byte().cpu().numpy()

def srgan_inference(image_path, output_path):
    """Perform inference on a single image and save the super-resolved image."""
    # Load the image
    img = Image.open(image_path).convert('L') 
    img_lr = img.resize((64, 64), Image.BICUBIC)  # Resized to low-res input size

    # Transform image to tensor
    img_tensor = transform(img_lr).unsqueeze(0).to(device)  # Add batch dimension

    # Generate SR image
    with torch.no_grad():
        sr_output = model(img_tensor)
        if isinstance(sr_output, tuple):
            sr_img_tensor = sr_output[0]  # Extract SR image if tuple
        else:
            sr_img_tensor = sr_output

    # Denormalize and convert back to image
    sr_img_np = denormalize(sr_img_tensor.squeeze(0).squeeze(0))
    sr_img = Image.fromarray(sr_img_np)

    # Save SR image
    sr_img.save(output_path)
    print(f"Super-resolved image saved to {output_path}")

# Example Usage
input_image = 'NIH_ChestXray_LR\test_image.png'
output_image = 'SRGAN_medical_imaging\outputs\super_resolved_test_image.png'

srgan_inference(input_image, output_image)
