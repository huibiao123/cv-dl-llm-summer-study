import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt 

torch.manual_seed(0)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Device:", device)

OUTPUT_DIR = Path(
    "./deep_learning/output"
)
OUTPUT_DIR.mkdir(
    exist_ok=True
)

transform = transforms.Compose([
    transforms.Resize(
        (64,64)
    ),
    
    transforms.Grayscale(
        num_output_channels=3
    ),
    
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

full_train_dataset = datasets.MNIST(
    root="./deep_learning/data",
    train=True,
    download=True,
    transform=transform
)

small_train_dataset, _ = random_split(
    full_train_dataset,
    [
        10000,
        len(full_train_dataset)-10000
    ],
    generator=torch.Generator()
    .manual_seed(0)
)

test_dataset = datasets.MNIST(
    root="./deep_learning/data",
    train=False,
    download=True,
    transform=transform
)

train_size = int(
    0.9 * len(small_train_dataset)
)

val_size = (
    len(small_train_dataset)
    -
    train_size
)

train_dataset, val_dataset = random_split(
    small_train_dataset,
    [
        train_size,
        val_size
    ],
    generator=torch.Generator()
    .manual_seed(0)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)

model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    10
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(
            outputs,
            labels
        )
        loss.backward()
        optimizer.step()
        total_loss += (
            loss.item()
            *
            images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )
        total += labels.size(0)
        correct += (
            predicted == labels
        ).sum().item()
    return (     
        total_loss / total,
        correct / total
    )

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                *
                images.size(0)
            )
            _, predicted = torch.max(
                outputs,
                1
            )
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum().item()
    return (
        total_loss / total,
        correct / total
    )

epochs = 5

history = {
    "epoch":[],
    "train_loss":[],
    "train_acc":[],
    "val_loss":[],
    "val_acc":[]
}

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate(
        val_loader
    )
    history["epoch"].append(
        epoch+1
    )
    history["train_loss"].append(
        train_loss
    )
    history["train_acc"].append(
        train_acc
    )
    history["val_loss"].append(
        val_loss
    )
    history["val_acc"].append(
        val_acc
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss:{train_loss:.4f} "
        f"Train Acc:{train_acc*100:.2f}% "
        f"Val Acc:{val_acc*100:.2f}%"
    )

test_loss, test_acc = evaluate(
    test_loader
)

print("\nFinal Test Accuracy:")
print(
    f"{test_acc*100:.2f}%"
)

torch.save(
    model.state_dict(),
    OUTPUT_DIR /
    "resnet18_mnist_transfer.pth"
)

df = pd.DataFrame(
    history
)

df.to_csv(
    OUTPUT_DIR /
    "resnet18_transfer_history.csv",
    index=False
)

plt.figure(
    figsize=(8,5)
)

plt.plot(
    history["epoch"],
    history["train_acc"],
    label="Train Accuracy"
)
plt.plot(
    history["epoch"],
    history["val_acc"],
    label="Validation Accuracy"
)
plt.xlabel(
    "Epoch"
)
plt.ylabel(
    "Accuracy"
)
plt.title(
    "ResNet18 Transfer Learning on MNIST"
)
plt.legend()
plt.savefig(
    OUTPUT_DIR /
    "resnet18_accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()