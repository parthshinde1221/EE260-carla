import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import timm
from sklearn.model_selection import train_test_split
import numpy as np
import gc
import os

torch.cuda.empty_cache()

# Constants
MEMORY_FRACTION = 0.6
EPOCHS = 10
MODEL_NAME = "Xception_ViT"
TRAINING_BATCH_SIZE = 16
EMBED_DIM = 768  # Dimension of the ViT model's input

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class NpyDataset(Dataset):
    def __init__(self, directories, start_frame=0):
        self.inputs = []
        self.outputs = []
        self.start_frame = start_frame
        
        # Load all data into memory from all directories
        for directory in directories:
            print(f'{directory} is being batched')
            inputs_file_path = os.path.join(directory, "inputs.npy")
            outputs_file_path = os.path.join(directory, "outputs.npy")

            inputs_file = open(inputs_file_path, "br")
            outputs_file = open(outputs_file_path, "br")

            while True:
                try:
                    input_data = np.load(inputs_file)
                    self.inputs.append(input_data)
                except:
                    break

            while True:
                try:
                    output_data = np.load(outputs_file)
                    self.outputs.append(output_data)
                except:
                    break

            inputs_file.close()
            outputs_file.close()

        self.inputs = np.array(self.inputs)[self.start_frame:]
        self.outputs = np.array(self.outputs)[self.start_frame:]

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]

        input_data = torch.tensor(input_data).float().permute(2, 0, 1)
        output_data = torch.tensor(output_data).float()

        return input_data, output_data

# Specify root directory containing all the subdirectories with input and output data
root_directory = "training_data_test_final_proj_3"
subdirectories = [os.path.join(root_directory, sub_dir) for sub_dir in os.listdir(root_directory)]

# Create dataset and dataloader
dataset = NpyDataset(subdirectories, start_frame=400)

# Splitting data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=False, pin_memory=True)

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

model = CustomXceptionViT()

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training setup with TensorBoard
writer = SummaryWriter(log_dir='./logs')

def train(model, loader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(loader) + batch_idx)
            torch.cuda.empty_cache()
            gc.collect()
        print(f'Epoch {epoch} ==> Loss: {loss.item()}')
    print('Training complete')

train(model, train_loader, EPOCHS)

# Evaluation
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(loader)

test_loss = evaluate(model, test_loader)
print(f'Test loss: {test_loss}')

# Save the model
model_file = f"testing_{MODEL_NAME}_2.pt"
torch.save(model.state_dict(), model_file)
print(f'Model saved to {model_file}')
