from torchvision import datasets, transforms

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
len(train_data), len(test_data)
# dataloadee

from torch.utils.data import DataLoader

train_datalaoder = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

len(train_datalaoder), len(test_dataloader)

# build a model
import torch
import torch.nn as nn

class AutoEncoderWithClassfication(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU()
        )
        self.classify = nn.Sequential(
            nn.Linear(16, 10)
        )

        self.decode = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )
    def forward(self,x):
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec, self.classify(enc)
    
dummyimage = torch.rand((32,1,28,28))
model = AutoEncoderWithClassfication()
# model(dummyimage)

loss_img = nn.MSELoss()
loss_class = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    return (correct/len(y_pred))*100

# train loop
from tqdm.auto import tqdm
epochs = 10

for epoch in tqdm(range(epochs), leave=False):
    model.train()
    for img, label in tqdm(train_datalaoder):
        img = img.flatten(start_dim = 1)
        y_img, y_pred = model(img)
        loss1 = loss_class(y_pred, label)
        loss2 = loss_img(y_img, img)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item():.4f}")

acc = 0
with torch.inference_mode():
    for img, label in tqdm(test_dataloader):
        img = img.flatten(start_dim = 1)
        y_img, y_pred = model(img)
        acc += accuracy_fn(y_pred.argmax(dim=1), label)
    acc /= len(test_dataloader)
    print(f"Accuracy : {acc:.4f}")