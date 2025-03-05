import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# =======================
# 1. Device Configuration
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 2. Data Loading (MNIST)
# =======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for better GAN stability
])

train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# =======================
# 3. Define Generator & Discriminator
# =======================

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # Outputs between -1 and 1 (match input normalization)
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)  # Reshape to image format

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability (0 to 1)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        return self.model(x)

# Instantiate models
noise_dim = 100
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# =======================
# 4. Define Loss & Optimizers
# =======================
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
lr = 0.0002

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# =======================
# 5. Training Loop
# =======================
epochs = 50
fixed_noise = torch.randn(16, noise_dim, device=device)  # Fixed noise for visualization

for epoch in range(epochs):
    for real_imgs, _ in train_dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # ---- Train Discriminator ----
        optimizer_D.zero_grad()
        
        # Real images (label = 1)
        real_labels = torch.ones(batch_size, 1, device=device)
        real_outputs = discriminator(real_imgs)
        d_loss_real = criterion(real_outputs, real_labels)

        # Fake images (label = 0)
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_imgs = generator(noise)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        fake_outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # ---- Train Generator ----
        optimizer_G.zero_grad()

        # Generate fake images & classify them
        fake_outputs = discriminator(fake_imgs)  # No detach() - we want gradients
        g_loss = criterion(fake_outputs, real_labels)  # Trick discriminator (want 1)

        g_loss.backward()
        optimizer_G.step()

    # Print progress
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

    # Generate and visualize images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            generated_imgs = generator(fixed_noise).cpu()
        
        fig, axes = plt.subplots(4, 4, figsize=(5, 5))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_imgs[i].squeeze(), cmap="gray")
            ax.axis("off")
        plt.show()

# =======================
# 6. Generate New Images
# =======================
with torch.no_grad():
    random_noise = torch.randn(16, noise_dim, device=device)
    generated_imgs = generator(random_noise).cpu()

fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated_imgs[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.show()
