import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(num_classes=10):

    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model
