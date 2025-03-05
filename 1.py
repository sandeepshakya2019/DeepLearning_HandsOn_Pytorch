import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class FashionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 10)
                )

    def forward(self, x):
        return self.layer(x)

model = FashionModel()
d = torch.randn((32,1,28,28))
#print(model(d).shape)

train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root="./data", train=False, download=False, transform=transforms.ToTensor())

train_dataloader = DataLoader(dataset = train_data, batch_size =32, shuffle=True )
test_dataloader = DataLoader(dataset = test_data, batch_size=32, shuffle=False)

print(len(train_dataloader), len(test_dataloader))

epochs = 1

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for i in range(epochs):
    model.train()
    total_loss = 0
    for img, label in train_dataloader:
        img = img.flatten(start_dim=1)
        y_pred = model(img)
        loss = loss_fn(y_pred, label)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss /= len(train_dataloader)
    print(f"Epoch : {i}, Loss : {total_loss}")

def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    return (correct/len(y_pred))*100

model.eval()

with torch.inference_mode():
    acc = 0
    for img, label in test_dataloader:
        img = img.flatten(start_dim=1)
        y_pred = model(img)
        acc += accuracy_fn(y_pred.argmax(dim=1),label)
    acc /= len(test_dataloader)
    print("Accuracy ", acc)

model.eval()

with torch.inference_mode():
    for img, label in test_dataloader:
        img = img.flatten(start_dim=1)
        y_pred = model(img)
        title = train_data.classes[y_pred.argmax(dim=1)]
        plt.imshow(img.view(1,28,28).squeeze())
        plt.title(title)
        plt.axis(False)
        plt.show()
        break
    
























