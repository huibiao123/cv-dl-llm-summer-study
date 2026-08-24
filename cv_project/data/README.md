# Fashion-MNIST Dataset

Fashion-MNIST 是一个用于图像分类任务的公开数据集，  
包含不同类型的服装和鞋类灰度图像。

本项目使用 Fashion-MNIST 进行服装图像分类，  
并使用 CNN 模型完成分类实验。

## 数据集基本信息

|项目|内容|
|---|---|
|数据集|Fashion-MNIST|
|训练集|60,000 张|
|测试集|10,000 张|
|图像大小|28 × 28|
|图像类型|灰度图|
|通道数|1|
|类别数量|10|
|任务类型|图像分类|

## 类别

|Label|Class|
|---|---|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

## 数据划分
原始训练集进一步划分为：
- Train：50,000 张
- Validation：10,000 张
- Test：10,000 张
其中：
- Train：用于模型训练
- Validation：用于训练过程中的模型评价
- Test：用于训练完成后的最终评价
## 数据预处理
原始图像首先通过 `ToTensor()` 转换为 PyTorch Tensor，  
并将像素值从 `[0, 255]` 转换到 `[0, 1]`。
随后使用：
```
transforms.Normalize(
    (0.5,),
    (0.5,)
)
```
进行标准化，使输入数据大致映射到 `[-1, 1]`。
最终输入 CNN 的数据形状为：
```
[1, 28, 28]
```
## 数据下载
数据集由 `torchvision.datasets.FashionMNIST`  
自动下载和加载，无需手动准备原始数据文件。
具体的数据加载和预处理过程见项目中的：
`dataset.py`
