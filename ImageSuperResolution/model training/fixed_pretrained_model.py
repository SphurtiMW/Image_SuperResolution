import torch

# Define paths
pretrained_model_path = 'pretrained_models\SRGAN.pt'
fixed_model_path = 'SRGAN_medical_imaging\fixed_SRGAN.pt'

# Load the pretrained model state dictionary
pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')

# Fix the input layer weight (change 3 input channels to 1 by averaging the RGB channels)
if 'conv01.body.0.weight' in pretrained_dict:
    pretrained_dict['conv01.body.0.weight'] = pretrained_dict['conv01.body.0.weight'].mean(dim=1, keepdim=True)
    print("Fixed input layer: conv01.body.0.weight")

# Fix the output layer weight (change output channels from 3 to 1 by summing)
if 'last_conv.body.0.weight' in pretrained_dict:
    pretrained_dict['last_conv.body.0.weight'] = pretrained_dict['last_conv.body.0.weight'].sum(dim=0, keepdim=True)
    pretrained_dict['last_conv.body.0.bias'] = pretrained_dict['last_conv.body.0.bias'].sum().unsqueeze(0)
    print("Fixed output layer: last_conv.body.0.weight and bias")

# Save the updated model weights
torch.save(pretrained_dict, fixed_model_path)

print(f"Modified weights saved to: {fixed_model_path}")
