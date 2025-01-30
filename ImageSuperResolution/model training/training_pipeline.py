import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

# Set device to CPU explicitly
device = torch.device("cpu")
print("Training on:", device)

# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Load dataset
class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, high_res_dir, transform=None):
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(high_res_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.high_res_dir, self.image_files[idx])
        hr_img = Image.open(img_path).convert('L')  # Convert to grayscale
        hr_img = hr_img.resize((256, 256), Image.BICUBIC)
        lr_img = hr_img.resize((64, 64), Image.BICUBIC)

        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img

# Load dataset
dataset_path = r'C:\Users\sphur\Downloads\NIH_ChestXray_LR'
dataset = MedicalImageDataset(dataset_path, transform)

# Split dataset (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader for training and validation
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Load the SRGAN generator model
from srgan_model import Generator  # Ensure srgan_model.py is in the working directory

model = Generator().to(device)

# Load and fix pretrained model weights
pretrained_model_path = r'C:\Users\sphur\Downloads\SRGAN_medical_imaging\fixed_SRGAN.pt'
pretrained_dict = torch.load(pretrained_model_path, map_location=device)

# Fix channel size mismatches (3 -> 1 conversion for grayscale images)
pretrained_dict['conv01.body.0.weight'] = pretrained_dict['conv01.body.0.weight'].mean(dim=1, keepdim=True)
pretrained_dict['last_conv.body.0.weight'] = pretrained_dict['last_conv.body.0.weight'].sum(dim=0, keepdim=True)
pretrained_dict['last_conv.body.0.bias'] = pretrained_dict['last_conv.body.0.bias'].sum().unsqueeze(0)

# Load the fixed weights into the model
model.load_state_dict(pretrained_dict)
print("Fixed pretrained model loaded successfully!")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for lr_imgs, hr_imgs in train_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        optimizer.zero_grad()
        
        # Ensure only the SR image is returned from the model
        if isinstance(model(lr_imgs), tuple):
            sr_imgs, _ = model(lr_imgs)  # Extract SR image, ignore auxiliary output
        else:
            sr_imgs = model(lr_imgs)

        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Fine-tuning complete!")

# Save the fine-tuned model
fine_tuned_model_path = r'C:\Users\sphur\Downloads\SRGAN_medical_imaging\fine_tuned_SRGAN.pt'
torch.save(model.state_dict(), fine_tuned_model_path)
print(f"Fine-tuned model saved successfully at: {fine_tuned_model_path}")
