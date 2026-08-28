import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_transfer_model(num_classes=10):

    # =========================
    # 加载 ImageNet 预训练 ResNet18
    # =========================

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )


    # =========================
    # 冻结特征提取层
    # =========================

    for param in model.parameters():

        param.requires_grad = False


    # =========================
    # 替换最后分类层
    # =========================

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes
    )


    return model