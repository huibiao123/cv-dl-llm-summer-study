import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from configs.config import (
    DATA_DIR,
    BATCH_SIZE,
    SEED,
    TRAIN_SIZE,
    VAL_SIZE
)


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# =========================
# ImageNet 预训练模型的输入处理
# =========================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_datasets():

    generator = torch.Generator()

    generator.manual_seed(SEED)


    full_train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform
    )


    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform
    )


    train_dataset, val_dataset, _ = random_split(
        full_train_dataset,
        [
            TRAIN_SIZE,
            VAL_SIZE,
            len(full_train_dataset)
            - TRAIN_SIZE
            - VAL_SIZE
        ],
        generator=generator
    )


    return (
        train_dataset,
        val_dataset,
        test_dataset
    )
def get_dataloaders():

    train_dataset, val_dataset, test_dataset = get_datasets()


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
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