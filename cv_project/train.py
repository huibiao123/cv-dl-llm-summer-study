import os
import csv

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import FashionCNN


# =========================
# 读取Config文件
# =========================

from configs.config import (
    SEED,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    MODEL_PATH,
    OUTPUT_DIR,
    HISTORY_PATH,
    LOSS_CURVE_PATH,
    ACCURACY_CURVE_PATH,
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================
# 设备选择
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)


# =========================
# 随机数种子
# =========================

torch.manual_seed(SEED)


# =========================
# 训练集载入
# =========================

train_loader, val_loader, _ = get_dataloaders()


# =========================
# 创建模型
# =========================

model = FashionCNN(
    num_classes=NUM_CLASSES
).to(device)


# =========================
# 定义损失函数
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# 优化器
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# 记录数据
# =========================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}


# =========================
# 开始训练
# =========================

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Forward
        outputs = model(images)


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # Backward
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        correct / total
    )


    # =====================
    # 验证
    # =====================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss / len(val_loader)
    )

    val_accuracy = (
        val_correct / val_total
    )


    # =====================
    # 数据记录
    # =====================

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    # =====================
    # 输出
    # =====================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # =====================
    # 保存最佳模型
    # =====================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best model saved. "
            f"Val Acc: {best_val_accuracy:.4f}"
        )


# =========================
# 保存到csv中
# =========================

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy"
        ]
    )


    for i in range(EPOCHS):

        writer.writerow(
            [
                i + 1,
                history["train_loss"][i],
                history["val_loss"][i],
                history["train_accuracy"][i],
                history["val_accuracy"][i]
            ]
        )


# =========================
# 绘制loss曲线
# =========================

epochs = range(
    1,
    EPOCHS + 1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Train Loss"
)


plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Fashion-MNIST Loss Curve"
)

plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(
    LOSS_CURVE_PATH,
    dpi=300
)


plt.close()


# =========================
# 绘制Accuracy曲线
# =========================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    epochs,
    history["train_accuracy"],
    marker="o",
    label="Train Accuracy"
)


plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Fashion-MNIST Accuracy Curve"
)

plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(
    ACCURACY_CURVE_PATH,
    dpi=300
)


plt.close()


# =========================
# 最终结果输出
# =========================

print()

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)
