import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


DATA_DIR = "./deep_learning/data"

BATCH_SIZE = 128
TRAIN_SAMPLES = 10000
VAL_RATIO = 0.1

SEED = 0


def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),

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


def get_datasets():

    transform = get_transform()

    full_train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    small_train_dataset, _ = random_split(
        full_train_dataset,
        [
            TRAIN_SAMPLES,
            len(full_train_dataset) - TRAIN_SAMPLES
        ],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_size = int(
        (1 - VAL_RATIO) * len(small_train_dataset)
    )

    val_size = (
        len(small_train_dataset)
        - train_size
    )

    train_dataset, val_dataset = random_split(
        small_train_dataset,
        [
            train_size,
            val_size
        ],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


def get_dataloaders():

    train_dataset, val_dataset, test_dataset = \
        get_datasets()

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )