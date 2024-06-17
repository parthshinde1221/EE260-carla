import torch
import numpy as np
import carla  # Import the CARLA simulator API
from carla import VehicleControl  # Assuming these are the correct class names; please adjust as necessary.
from agents.navigation.basic_agent import BasicAgent
# from model_new import CustomXception
# from model_vit import 
import torch
import torch.nn as nn
import timm

EMBED_DIM = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

class CustomXception(nn.Module):
    def __init__(self):
        super(CustomXception, self).__init__()
        self.base_model = timm.create_model('xception', pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 3)
        self.to(device)

    def forward(self, x):
        return self.base_model(x)
    

# Model definition
class CustomXceptionViT(nn.Module):
    def __init__(self):
        super(CustomXceptionViT, self).__init__()
        self.xception = timm.create_model('xception', pretrained=True)
        self.xception.fc = nn.Identity()  # Remove the fully connected layer

        self.projection = nn.Linear(2048, EMBED_DIM)  # Projecting to the ViT's embedding dimension

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.patch_embed = nn.Identity()  # Remove the patch embedding
        self.vit.head = nn.Linear(self.vit.head.in_features, 3)  # Adjust the final layer to match the number of outputs

        self.to(device)

    def forward(self, x):
        # Extract unpooled features from Xception
        x = self.xception.forward_features(x)  # Shape: [B, 2048, 7, 7]
        
        # Flatten the spatial dimensions and project to the Transformer dimension
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # Shape: [B, 49, 2048]
        x = self.projection(x)  # Shape: [B, 49, EMBED_DIM]

        # Add classification token and positional embeddings
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # Shape: [B, 1, EMBED_DIM]
        x = torch.cat((cls_token, x), dim=1)  # Shape: [B, 50, EMBED_DIM]
        x = x + self.vit.pos_embed[:, :(x.size(1)), :]  # Shape: [B, 50, EMBED_DIM]

        # Pass through the Transformer
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        x = self.vit.head(x[:, 0])  # Use the classification token output
        return x

class ModelAgent(BasicAgent):  # Ensure this inherits from the correct CARLA Agent class.
    def __init__(self,vehicle,model_path):
        super().__init__(vehicle)  # Initialize the superclass if necessary.
        # self.model = CustomXception()
        self.model = CustomXceptionViT()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Model loaded and set to evaluation mode.")

    @staticmethod
    def process_image(image):
        # Assuming image comes in with dimensions [H, W, C] and includes an alpha channel.
        processed_image = image[:, :, :3]  # Discard the alpha channel if present.
        return processed_image

    def run_step(self,sensor_data,measurements=1,directions=1,target=1):
        current_speed = measurements
        input_image = self.process_image(sensor_data)  # Adjust key according to actual data structure.
        input_image = input_image / 255.0  # Normalize the image data.
        input_tensor = torch.tensor(input_image).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()

        with torch.no_grad():
            control_output = self.model(input_tensor)
            steer, throttle, brake = control_output[0]  # Unpack output assuming model outputs in this order.

        control = VehicleControl()
        control.throttle = throttle.item()
        control.steer = steer.item()
        # control.brake = brake.item()

        # Speed control logic to avoid speeding
        # if current_speed > 10.0 and control.brake < 0.1:
        #     control.throttle = max(0.0, 1.0 - current_speed / 35.0)

        control.hand_brake = False
        control.reverse = False
        return control

print('Initialization complete. Ready to run the agent.')
