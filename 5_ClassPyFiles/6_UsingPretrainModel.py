import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models

# Function to compute accuracy using argmax
def accuracy(y_pred, y_true):
    predicted = torch.argmax(y_pred, dim=1)  # Use argmax to get predicted labels
    correct = (predicted == y_true).sum().item()
    return (correct / y_true.size(0)) * 100  # Convert to percentage

# Select a model (Change 'resnet18' to another model if needed)
MODEL_NAME = "resnet18"
device = torch.device("cpu")

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
print(f"\nLoading model: {MODEL_NAME}")

# Modify the final layer for MNIST (10 classes)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)

# Move model to device and set to evaluation mode
model = model.to(device)
model.eval()

# Define transformations for MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.Grayscale(3),        # Convert grayscale to 3-channel
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load MNIST test dataset
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate model on MNIST
print("Running in prediction mode...")
total_correct, total_samples = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Compute batch accuracy
        batch_acc = accuracy(outputs, labels)
        total_correct += (batch_acc / 100) * labels.size(0)
        total_samples += labels.size(0)

# Compute final accuracy
final_accuracy = (total_correct / total_samples) * 100
print(f"Accuracy of {MODEL_NAME} on MNIST test data: {final_accuracy:.2f}%")
