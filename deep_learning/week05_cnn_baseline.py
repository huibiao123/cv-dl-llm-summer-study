import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

torch.manual_seed(0)

OUTPUT_DIR = Path(
    "./deep_learning/output"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

full_train_dataset = datasets.MNIST(
    root="./deep_learning/data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./deep_learning/data",
    train=False,
    download=True,
    transform=transform
)

train_dataset,val_dataset = random_split(
    full_train_dataset,
    [50000,10000]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64
)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64*7*7,
                128
            ),
            nn.ReLU(),
            nn.Dropout(
                0.3
            ),
            nn.Linear(
                128,
                10
            )
        )

    def forward(self,x):
        x=self.conv(x)
        x=self.fc(x)
        return x

model=CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs=10

train_loss_history=[]

val_loss_history=[]

for epoch in range(epochs):
    model.train()
    total_loss=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        output=model(images)
        loss=criterion(
            output,
            labels
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = (
        total_loss /
        len(train_loader)
    )

    model.eval()
    val_loss=0
    with torch.no_grad():
        for images,labels in val_loader:
            images=images.to(device)
            labels=labels.to(device)
            output=model(images)
            loss=criterion(
                output,
                labels
            )
            val_loss += loss.item()
    val_loss /= len(val_loader)
    train_loss_history.append(
        train_loss
    )
    val_loss_history.append(
        val_loss
    )
    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train:{train_loss:.4f} "
        f"Val:{val_loss:.4f}"
    )

plt.figure(
    figsize=(8,5)
)
plt.plot(
    train_loss_history,
    label="Train"
)
plt.plot(
    val_loss_history,
    label="Validation"
)
plt.xlabel(
    "Epoch"
)
plt.ylabel(
    "Loss"
)
plt.legend()
plt.savefig(
    OUTPUT_DIR /
    "cnn_loss.png"
)
plt.close()

model.eval()
correct=0
total=0
images_all=[]
labels_all=[]
preds_all=[]
with torch.no_grad():
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        output=model(images)
        preds=torch.argmax(
            output,
            dim=1
        )
        correct += (
            preds==labels
        ).sum().item()
        total += labels.size(0)
        images_all.extend(
            images.cpu()
        )
        labels_all.extend(
            labels.cpu()
        )
        preds_all.extend(
            preds.cpu()
        )
accuracy = (
    correct/total*100
)
print(
    f"Test Accuracy:{accuracy:.2f}%"
)


def visualize(
    index_list,
    name,
    title
):
    plt.figure(
        figsize=(10,5)
    )
    for i,index in enumerate(index_list):
        plt.subplot(
            2,
            5,
            i+1
        )
        plt.imshow(
            images_all[index].squeeze(),
            cmap="gray"
        )
        plt.title(
            f"T:{labels_all[index]}\nP:{preds_all[index]}"
        )
        plt.axis(
            "off"
        )
    plt.suptitle(title)
    plt.savefig(
        OUTPUT_DIR/name
    )
    plt.close()

correct_index=[]
for i in range(len(labels_all)):
    if labels_all[i]==preds_all[i]:
        correct_index.append(i)
    if len(correct_index)==10:
        break
visualize(
    correct_index,
    "cnn_prediction_samples.png",
    "Correct Predictions"
)

error_index=[]
for i in range(len(labels_all)):
    if labels_all[i]!=preds_all[i]:
        error_index.append(i)
    if len(error_index)==10:
        break
visualize(
    error_index,
    "cnn_error_samples.png",
    "Wrong Predictions"
)

pd.DataFrame(
    {
        "Model":[
            "CNN"
        ],
        "Accuracy":[
            accuracy
        ]
    }
).to_csv(
    OUTPUT_DIR/
    "cnn_result.csv",
    index=False
)
