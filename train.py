import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Load settings from YAML
def load_config(file_path="settings.yaml"):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Get data loader
def get_dataloader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root=config["paths"]["data_dir"], train=True, transform=transform, download=True)
    return DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

# Train function
def train():
    train_loader = get_dataloader()

    # Initialize model, loss, optimizer
    model = SimpleMLP(config["model"]["input_size"], config["model"]["hidden_size"], config["model"]["output_size"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {loss.item():.4f}")

    # Save model
    os.makedirs(os.path.dirname(config["paths"]["model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["paths"]["model_save_path"])
    print("Model saved successfully!")

# Run training
if __name__ == "__main__":
    train()
