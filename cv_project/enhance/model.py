import torch
import torch.nn as nn


class FashionCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            # [1, 28, 28]
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1                                                  #维持尺寸不变
            ),

            # [32, 28, 28]
            nn.ReLU(),

            # [32, 14, 14]
            nn.MaxPool2d(
                kernel_size=2
            ),


            # [32, 14, 14]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            # [64, 14, 14]
            nn.ReLU(),

            # [64, 7, 7]
            nn.MaxPool2d(
                kernel_size=2
            )
        )


        self.classifier = nn.Sequential(

            # [64, 7, 7]
            # -> [3136]
            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":                                           #自检和展示

    model = FashionCNN()

    print(model)                                                     #打印网络结构


    # 模拟一个 batch
    x = torch.randn(                                                 #随机生成一个 Tensor，模拟真实输入。
        64,
        1,
        28,
        28
    )


    output = model(x)                                                #PyTorch 自动调用 forward()


    print(                                                           #展示输入/输出的张量
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )