from torchvision import datasets
from torchvision import transforms

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

len(train_data), len(test_data)

class_names = train_data.classes

import matplotlib.pyplot as plt
for img, label in train_data:
    print(f"Shape of Img", img.shape)
    plt.figure(figsize=(2,2))
    plt.imshow(img.squeeze())
    plt.title(class_names[label])
    plt.axis(False)
    plt.savefig("filename.png")
    break

from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

len(train_dataloader), len(test_dataloader)

for batch_idx, (img, label) in enumerate(train_dataloader):
    print(f"Batch indx {batch_idx} Images shape {img.shape} label shapoe {label.shape}")
    break

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Linear(in_features=16, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

import torch
dummy_image = torch.rand((32,1,28,28))
model = AutoEncoder(28*28)
# model(dummy_image)
from torchinfo import summary

summary(model, input_size=(32,1,28,28))

# loss_fn = nn.CrossEntropyLoss()
# since output is an image itself aso use mseloss
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

from tqdm.auto import tqdm
epochs = 10
for epoch in tqdm(range(epochs)):
    train_loss = 0
    model.train()
    for img, label in tqdm(train_dataloader, leave=False):
        img = img.flatten(start_dim=1)
        # print(img.shape)
        y_pred = model(img)
        loss = loss_fn(y_pred, img)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    print(f"Loss: {loss.item():.4f}")

model.eval()
test_loss = 0
with torch.no_grad():  # No gradients needed during evaluation
    for img, _ in tqdm(test_dataloader, leave=False):
        img = img.flatten(start_dim=1) # Flatten images
        y_pred = model(img)
        loss = loss_fn(y_pred, img)  # Compare with original input
        test_loss += loss.item()

test_loss /= len(test_dataloader)
print(f"Test Loss: {test_loss:.4f}")

img, label = next(iter(test_dataloader))
img = img.flatten(start_dim=1)

# Pass through the model
with torch.no_grad():
    reconstructed = model(img)

img = img.view(-1, 1, 28, 28)
reconstructed = reconstructed.view(-1, 1, 28, 28)

for i in range(6):
    print("--------------")
    random_idx = torch.randint(0,32,size=[1])
    getorgimg = img[random_idx]
    getreimg = reconstructed[random_idx]

    plt.figure(figsize=(2,2))
    plt.imshow(getorgimg.squeeze())
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(2,2))
    plt.imshow(getreimg.squeeze())
    plt.title("Reconstru Image")
    plt.axis("off")
    plt.show()
