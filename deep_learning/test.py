import torch
import matplotlib.pyplot as plt

from pathlib import Path

from dataset import get_dataloaders
from model import get_model


OUTPUT_DIR = Path(
    "./deep_learning/output"
)

MODEL_PATH = (
    OUTPUT_DIR / "best_model.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def test_model(
    model,
    loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    confusion_matrix = torch.zeros(
        10,
        10,
        dtype=torch.int64
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            for true, pred in zip(
                labels.cpu(),
                predicted.cpu()
            ):

                confusion_matrix[
                    true,
                    pred
                ] += 1

    accuracy = correct / total

    return accuracy, confusion_matrix


def plot_confusion_matrix(
    matrix
):

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        matrix.numpy()
    )

    plt.colorbar()

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xticks(
        range(10)
    )

    plt.yticks(
        range(10)
    )

    for i in range(10):

        for j in range(10):

            plt.text(
                j,
                i,
                str(matrix[i, j].item()),
                ha="center",
                va="center"
            )

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    train_loader, val_loader, test_loader = \
        get_dataloaders()

    model = get_model(
        num_classes=10
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    test_acc, matrix = test_model(
        model,
        test_loader,
        DEVICE
    )

    print(
        f"Test Accuracy: {test_acc * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        matrix.numpy()
    )

    plot_confusion_matrix(
        matrix
    )


if __name__ == "__main__":
    main()