from flask import Flask, request, render_template, send_file
import os
import io
from PIL import Image
import torch
from torchvision import transforms
from srgan_model import Generator

# Initialize Flask app
app = Flask(__name__)

# Load the SRGAN model
device = torch.device("cpu")
model_path = r'C:\Users\sphur\Downloads\SRGAN_medical_imaging\fine_tuned_SRGAN.pt'  
model = Generator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Function to denormalize tensors
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # Scale to [0, 1]
    tensor = tensor * 255.0  # Scale to [0, 255]
    return tensor.clamp(0, 255).byte().cpu().numpy()

@app.route('/')
def home():
    return render_template('index.html')  # HTML page for user interaction

@app.route('/super_resolve', methods=['POST'])
def super_resolve():
    if 'file' not in request.files:
        return "No file uploaded", 400

    # Get the uploaded image
    file = request.files['file']
    img = Image.open(file.stream).convert('L')  # Convert to grayscale
    img_lr = img.resize((64, 64), Image.BICUBIC)  # Resize to low-res size

    # Transform and run inference
    img_tensor = transform(img_lr).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_output = model(img_tensor)
        if isinstance(sr_output, tuple):
            sr_img_tensor = sr_output[0]  # Extract SR image if tuple
        else:
            sr_img_tensor = sr_output

    # Convert to image format
    sr_img_np = denormalize(sr_img_tensor.squeeze(0).squeeze(0))
    sr_img = Image.fromarray(sr_img_np)

    # Save image to in-memory file
    img_io = io.BytesIO()
    sr_img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
