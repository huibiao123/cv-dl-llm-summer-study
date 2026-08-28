import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# =========================
# Config引用
# =========================

from configs.config import (
    DATA_DIR,
    BATCH_SIZE,
    SEED,
    TRAIN_SIZE,
    VAL_SIZE
)


# =========================
# 分类名称
# =========================

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
# 数据预处理，先转张量然后缩放最后标准化
# =========================

transform = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ]
)


# =========================
# 下载数据集而且划分数据集
# =========================

def get_datasets():

    # 固定随机种子
    generator = torch.Generator()

    generator.manual_seed(SEED)


    # 原始训练集
    full_train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )


    # 测试集
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )


    # Train / Validation split

    train_dataset, val_dataset, _ = random_split(
    full_train_dataset,
    [
        TRAIN_SIZE,
        VAL_SIZE,
        len(full_train_dataset) - TRAIN_SIZE - VAL_SIZE
    ],
    generator=generator
)


    return (
        train_dataset,
        val_dataset,
        test_dataset
    )



# =========================
# DataLoader 构建
# =========================

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



# =========================
# dataset的测试（自检与参数展示）
# =========================

if __name__ == "__main__":                                         #只有单独运行这部分时才执行


    train_loader, val_loader, test_loader = get_dataloaders()


    print(
        "Train:",
        len(train_loader.dataset)
    )


    print(
        "Validation:",
        len(val_loader.dataset)
    )


    print(
        "Test:",
        len(test_loader.dataset)
    )


    images, labels = next(
        iter(train_loader)
    )


    print(
        "Image shape:",
        images.shape
    )


    print(
        "Label shape:",
        labels.shape
    )


    print(
        "Image range:",
        images.min().item(),
        images.max().item()
    )


    print(
        "Example label:",
        labels[0].item()
    )


    print(
        "Class:",
        class_names[
            labels[0].item()
        ]
    )