import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


torch.manual_seed(0)

OUTPUT_DIR = Path("./deep_learning/output")
OUTPUT_DIR.mkdir(
    exist_ok=True
)

transform = transforms.Compose([
    transforms.ToTensor()
])

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

train_size = 50000
val_size = 10000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [
        train_size,
        val_size
    ]
)

print(
    "训练集:",
    len(train_dataset)
)

print(
    "验证集:",
    len(val_dataset)
)

print(
    "测试集:",
    len(test_dataset)
)

class MLP(nn.Module):

    def __init__(
        self,
        hidden_size=128,
        dropout=0
    ):

        super().__init__()

        self.model = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                784,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                10
            )

        )

    def forward(self,x):

        return self.model(x)

def evaluate(
        model,
        loader,
        loss_function
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in loader:

            output = model(images)

            loss = loss_function(
                output,
                labels
            )

            total_loss += loss.item()

            prediction = torch.argmax(
                output,
                dim=1
            )

            correct += (
                prediction == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = (
        total_loss /
        len(loader)
    )

    accuracy = correct / total

    return avg_loss, accuracy

def train_model(
        name,
        hidden_size,
        lr,
        batch_size,
        dropout
):

    print("\n================")
    print(name)
    print("================")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1000
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000
    )

    model = MLP(
        hidden_size,
        dropout
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    epochs = 10

    train_loss_history = []

    val_loss_history = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for images,labels in train_loader:

            output = model(images)

            loss = loss_function(
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

        val_loss,val_acc = evaluate(
            model,
            val_loader,
            loss_function
        )

        train_loss_history.append(
            train_loss
        )

        val_loss_history.append(
            val_loss
        )

        print(
            f"Epoch {epoch+1}/{epochs}",
            f"Train Loss={train_loss:.4f}",
            f"Val Loss={val_loss:.4f}",
            f"Val Acc={val_acc:.4f}"
        )

    test_loss,test_acc = evaluate(
        model,
        test_loader,
        loss_function
    )

    print(
        "Test Accuracy:",
        test_acc
    )

    return (
        test_acc,
        train_loss_history,
        val_loss_history
    )

acc1,train1,val1 = train_model(
    name="实验A: MLP-128",
    hidden_size=128,
    lr=0.001,
    batch_size=64,
    dropout=0
)

acc2,train2,val2 = train_model(
    name="实验B: MLP-256-Dropout",
    hidden_size=256,
    lr=0.0005,
    batch_size=128,
    dropout=0.3
)

plt.figure(
    figsize=(8,5)
)

plt.plot(
    train1,
    label="MLP-128 Train"
)

plt.plot(
    val1,
    label="MLP-128 Validation"
)

plt.plot(
    train2,
    label="MLP-256-Dropout Train"
)

plt.plot(
    val2,
    label="MLP-256-Dropout Validation"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "MNIST MLP Train Validation Loss"
)

plt.legend()

plt.grid()

plt.savefig(
    OUTPUT_DIR /
    "train_validation_loss.png",
    dpi=300
)

plt.show()

result_table = pd.DataFrame({

    "Experiment":[
        "MLP-128",
        "MLP-256-Dropout"
    ],

    "Hidden_Size":[
        128,
        256
    ],

    "Learning_Rate":[
        0.001,
        0.0005
    ],

    "Batch_Size":[
        64,
        128
    ],

    "Dropout":[
        0,
        0.3
    ],

    "Test_Accuracy":[
        acc1,
        acc2
    ]

})

result_table.to_csv(
    OUTPUT_DIR /
    "accuracy_result.csv",
    index=False
)

print("\n实验结果")
print("----------------")

print(result_table)