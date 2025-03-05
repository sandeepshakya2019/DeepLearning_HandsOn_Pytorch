import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 28 * 28  # MNIST images are 28x28
hidden_dim = 128     # First hidden layer size
output_dim = 10      # Number of classes (0-9)
learning_rate = 0.01
epochs = 2000

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize weights and biases manually
w1 = torch.randn(input_dim, hidden_dim, requires_grad=True, device=device) * 0.01
b1 = torch.zeros(hidden_dim, requires_grad=True, device=device)

w2 = torch.randn(hidden_dim, hidden_dim, requires_grad=True, device=device) * 0.01
b2 = torch.zeros(hidden_dim, requires_grad=True, device=device)

w3 = torch.randn(hidden_dim, output_dim, requires_grad=True, device=device) * 0.01
b3 = torch.zeros(output_dim, requires_grad=True, device=device)

# Forward function
def forward(x):
    x = x.view(-1, input_dim)  # Flatten image
    z1 = x @ w1 + b1
    a1 = F.relu(z1)

    z2 = a1 @ w2 + b2
    a2 = F.relu(z2)

    z3 = a2 @ w3 + b3
    return F.softmax(z3, dim=1)  # Output probabilities

# Cross-entropy loss
def cross_entropy(pred, target):
    return -torch.sum(target * torch.log(pred + 1e-9)) / pred.shape[0]

# Training loop
for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # One-hot encode labels
        y_train = torch.eye(output_dim, device=device)[labels]

        # Forward pass
        y_pred = forward(images)
        loss = cross_entropy(y_pred, y_train)

        # Backpropagation
        loss.backward()
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            b1 -= learning_rate * b1.grad
            w2 -= learning_rate * w2.grad
            b2 -= learning_rate * b2.grad
            w3 -= learning_rate * w3.grad
            b3 -= learning_rate * b3.grad

            # Zero gradients
            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()
            w3.grad.zero_()
            b3.grad.zero_()

    # Print loss every 250 epochs
    if (epoch + 1) % 250 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test on a batch of images and visualize predictions
def test_and_visualize():
    model.eval()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    y_pred = forward(images)
    predictions = torch.argmax(y_pred, dim=1)

    # Convert images to numpy for visualization
    images = images.cpu().numpy()

    # Plot 6 images with predictions
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"True: {labels[i].item()}, Pred: {predictions[i].item()}")
        plt.axis("off")
    plt.show()

test_and_visualize()

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 28 * 28  # MNIST images are 28x28
hidden_dim = 128     # First hidden layer size
output_dim = 10      # Number of classes (0-9)
learning_rate = 0.01
epochs = 2000

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize weights and biases manually
w1 = torch.randn(input_dim, hidden_dim, requires_grad=True, device=device) * 0.01
b1 = torch.zeros(hidden_dim, requires_grad=True, device=device)

w2 = torch.randn(hidden_dim, hidden_dim, requires_grad=True, device=device) * 0.01
b2 = torch.zeros(hidden_dim, requires_grad=True, device=device)

w3 = torch.randn(hidden_dim, output_dim, requires_grad=True, device=device) * 0.01
b3 = torch.zeros(output_dim, requires_grad=True, device=device)

# Define predefined loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Forward function
def forward(x):
    x = x.view(-1, input_dim)  # Flatten image
    z1 = x @ w1 + b1
    a1 = F.relu(z1)

    z2 = a1 @ w2 + b2
    a2 = F.relu(z2)

    z3 = a2 @ w3 + b3
    return z3  # No softmax needed, as CrossEntropyLoss applies it internally

# Training loop
for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        y_pred = forward(images)
        loss = loss_fn(y_pred, labels)  # Using predefined loss function

        # Backpropagation
        loss.backward()
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            b1 -= learning_rate * b1.grad
            w2 -= learning_rate * w2.grad
            b2 -= learning_rate * b2.grad
            w3 -= learning_rate * w3.grad
            b3 -= learning_rate * b3.grad

            # Zero gradients
            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()
            w3.grad.zero_()
            b3.grad.zero_()

    # Print loss every 250 epochs
    if (epoch + 1) % 250 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test on a batch of images and visualize predictions
def test_and_visualize():
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    y_pred = forward(images)
    predictions = torch.argmax(y_pred, dim=1)

    # Convert images to numpy for visualization
    images = images.cpu().numpy()

    # Plot 6 images with predictions
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"True: {labels[i].item()}, Pred: {predictions[i].item()}")
        plt.axis("off")
    plt.show()

test_and_visualize()
