import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Generate Dataset using `make_blobs`
num_classes = 3
X, y = make_blobs(n_samples=1000, centers=num_classes, random_state=42, cluster_std=2.0)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = torch.tensor(X_train, dtype=torch.float32, device=device), torch.tensor(y_train, dtype=torch.long, device=device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32, device=device), torch.tensor(y_test, dtype=torch.long, device=device)

# 2️⃣ Define Model Parameters (without `nn.Linear`)
input_dim = X.shape[1]   # Number of features
hidden_dim = 32
output_dim = num_classes
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
w1 = torch.randn(input_dim, hidden_dim, requires_grad=True, device=device) * 0.01
b1 = torch.zeros(hidden_dim, requires_grad=True, device=device)
w2 = torch.randn(hidden_dim, hidden_dim, requires_grad=True, device=device) * 0.01
b2 = torch.zeros(hidden_dim, requires_grad=True, device=device)
w3 = torch.randn(hidden_dim, output_dim, requires_grad=True, device=device) * 0.01
b3 = torch.zeros(output_dim, requires_grad=True, device=device)

# 3️⃣ Define Forward Pass Function
def forward(x):
    z1 = x @ w1 + b1
    a1 = F.relu(z1)
    
    z2 = a1 @ w2 + b2
    a2 = F.relu(z2)
    
    z3 = a2 @ w3 + b3
    return z3  # No softmax, as CrossEntropyLoss applies it

# Define loss function
loss_fn = torch.nn.CrossEntropyLoss()

# 4️⃣ Training Loop
for epoch in range(epochs):
    y_pred = forward(X_train)  # Forward pass
    loss = loss_fn(y_pred, y_train)  # Compute loss

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

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5️⃣ Evaluate Model Accuracy
def accuracy(X, y):
    y_pred = forward(X)
    y_pred_class = torch.argmax(y_pred, dim=1)
    acc = (y_pred_class == y).float().mean().item() * 100
    return acc

train_acc = accuracy(X_train, y_train)
test_acc = accuracy(X_test, y_test)

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# 6️⃣ Visualize Decision Boundary
def plot_decision_boundary():
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid = torch.tensor(X_grid, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        Z = forward(X_grid)
        Z = torch.argmax(Z, dim=1).cpu().numpy()
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary()
