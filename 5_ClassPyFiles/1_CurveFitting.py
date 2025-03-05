import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
torch.manual_seed(42)  # For reproducibility
X = torch.linspace(-2, 2, 100).reshape(-1, 1)  # 100 points between -2 and 2
y = 3*X**3 - 2*X**2 + X + 5 + torch.randn_like(X) * 2  # y = 3x^3 - 2x^2 + x + 5 + noise

# Define a simple polynomial regression model
class PolyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1, requires_grad=True))  # Coefficient for x^3
        self.w2 = nn.Parameter(torch.randn(1, requires_grad=True))  # Coefficient for x^2
        self.w3 = nn.Parameter(torch.randn(1, requires_grad=True))  # Coefficient for x
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))   # Bias term

    def forward(self, x):
        return self.w1 * x**3 + self.w2 * x**2 + self.w3 * x + self.b

# Initialize model
model = PolyModel()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X)  # Forward pass
    loss = criterion(y_pred, y)  # Compute loss

    optimizer.zero_grad()  # Zero gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Plot results
plt.scatter(X, y, label="Original Data", color="blue", alpha=0.6)
plt.plot(X, model(X).detach(), label="Fitted Curve", color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Curve Fitting with PyTorch")
plt.legend()
plt.show()

# Print learned parameters
print(f"Learned Parameters: w1={model.w1.item():.3f}, w2={model.w2.item():.3f}, w3={model.w3.item():.3f}, b={model.b.item():.3f}")


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
torch.manual_seed(42)  # For reproducibility
X = torch.linspace(-2, 2, 100).reshape(-1, 1)  # 100 points between -2 and 2
y = 3*X**3 - 2*X**2 + X + 5 + torch.randn_like(X) * 2  # y = 3x^3 - 2x^2 + x + 5 + noise

# Create polynomial features (x, x^2, x^3)
X_poly = torch.cat([X**3, X**2, X], dim=1)  # Shape: (100, 3)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(3, 10),  # Input: (x^3, x^2, x), Output: 10 neurons
    nn.ReLU(),
    nn.Linear(10, 1)   # Output: 1 neuron (predict y)
)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_poly)  # Forward pass
    loss = criterion(y_pred, y)  # Compute loss

    optimizer.zero_grad()  # Zero gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Plot results
plt.scatter(X, y, label="Original Data", color="blue", alpha=0.6)
plt.plot(X, model(X_poly).detach(), label="Fitted Curve", color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Curve Fitting with PyTorch (Using nn.Sequential)")
plt.legend()
plt.show()
