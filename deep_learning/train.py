import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from dataset import get_dataloaders
from model import get_model


SEED = 0
EPOCHS = 5
LEARNING_RATE = 0.001

OUTPUT_DIR = Path(
    "./deep_learning/output"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger():

    logger = logging.getLogger(
        "training"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        OUTPUT_DIR / "training.log",
        mode="w",
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(
        formatter
    )

    console_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(message)s")
)

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )

    return logger


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

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
            * images.size(0)
        )

        predicted = outputs.argmax(
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy


def evaluate(
    model,
    loader,
    criterion,
    device
):

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
                * images.size(0)
            )

            predicted = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy


def plot_history(history):

    plt.figure(
        figsize=(8, 5)
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
        OUTPUT_DIR / "accuracy_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    set_seed(SEED)

    logger = setup_logger()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Device: {device}"
    )

    logger.info(
        f"Seed: {SEED}"
    )

    logger.info(
        f"Epochs: {EPOCHS}"
    )

    logger.info(
        f"Learning Rate: {LEARNING_RATE}"
    )

    train_loader, val_loader, test_loader = \
        get_dataloaders()

    logger.info(
        f"Train samples: {len(train_loader.dataset)}"
    )

    logger.info(
        f"Validation samples: {len(val_loader.dataset)}"
    )

    logger.info(
        f"Test samples: {len(test_loader.dataset)}"
    )

    model = get_model(
        num_classes=10
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_val_acc = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):

        train_loss, train_acc = \
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

        val_loss, val_acc = \
            evaluate(
                model,
                val_loader,
                criterion,
                device
            )

        history["epoch"].append(
            epoch + 1
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

        logger.info(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc * 100:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "best_model.pth"
            )

            logger.info(
                f"Best model saved. "
                f"Val Acc: {val_acc * 100:.2f}%"
            )

    df = pd.DataFrame(
        history
    )

    df.to_csv(
        OUTPUT_DIR / "history.csv",
        index=False
    )

    plot_history(
        history
    )

    logger.info(
        "Training finished."
    )

    logger.info(
        f"Best Validation Accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    main()