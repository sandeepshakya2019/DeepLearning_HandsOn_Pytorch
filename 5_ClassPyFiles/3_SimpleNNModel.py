import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define Fully Connected Neural Network using nn.Sequential
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output layer
        )

    def forward(self, X):
        return self.model(X)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define input, hidden, and output dimensions
input_dim = 28 * 28  # MNIST image size (flattened)
hidden_dim = 128
output_dim = 10  # 10 classes for MNIST

# Create model instance
model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(-1, 28 * 28)  # Flatten the images

        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Testing loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(-1, 28 * 28)  # Flatten images

        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Visualization: Show 6 random test images with predictions
def visualize_predictions(model, test_loader, device):
    model.eval()
    images, labels = next(iter(test_loader))  # Get a batch of test images
    images, labels = images.to(device), labels.to(device)
    images_flattened = images.view(-1, 28 * 28)  # Flatten images

    with torch.no_grad():
        outputs = model(images_flattened)
        predictions = outputs.argmax(dim=1)

    images = images.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for i in range(6):  # Show 6 random images
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"True: {labels[i].item()}, Pred: {predictions[i].item()}")
        plt.axis("off")
    plt.show()

visualize_predictions(model, test_loader, device)
