
# Fashion-MNIST CNN
基于 CNN 的 Fashion-MNIST 服装图像分类项目。
## 文件说明
### datasetpy
负责 Fashion-MNIST 数据集的加载、预处理和数据划分。
主要功能：
- 下载 Fashion-MNIST 数据集
- 将图像转换为 Tensor
- 对图像进行标准化
- 将训练集划分为 Train 和 Validation
- 创建 DataLoader
数据划分：
- Train：50,00
- Validation：10,00
- Test：10,000
运行：
```
python dataset.py
```
用于检查数据集是否能够正常加载，以及确认图像和标签的尺寸。
### model.py
负责定义 CNN 分类模型。
模型结构：
```
Input
  ↓
Conv2d
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2d
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear
  ↓
Linear
  ↓
10 Classes
```
输入图像：
```
[1, 28, 28]
```
模型输出：
```
[10]
```
运行：
```
python model.py
```
用于检查模型结构以及输入输出尺寸是否正确。
### train.py
负责模型训练和验证。
主要功能：
- 加载训练集和验证集
- 创建 CNN 模型
- 使用 CrossEntropyLoss 计算损失
- 使用 Adam 优化模型
- 记录 Train Loss 和 Validation Loss
- 记录 Train Accuracy 和 Validation Accuracy
- 保存 Validation Accuracy 最好的模型
运行：
```
python train.py
```
训练完成后，模型和训练记录保存在 `outputs/` 目录。
主要输出：
- `best_model.pth`
- `history.csv`
- `loss_curve.png`
- `accuracy_curve.png`

### test.py
负责使用测试集对训练完成的模型进行最终评价。
主要功能：
- 加载 `best_model.pth`
- 在 Test 数据集上进行推理
- 计算 Test Accuracy
- 生成 Confusion Matrix
运行：
```
python test.py
```
主要输出：
```
Test Accuracy
```
以及：
```
outputs/confusion_matrix.png
```
## 使用顺序
第一次运行项目时：
```
python dataset.py

python model.py

python train.py

python test.py
```
其中：
`dataset.py` 用于检查数据加载，
`model.py` 用于检查模型结构，
`train.py` 用于训练模型，
`test.py` 用于最终测试。
## 实验流程
```
Fashion-MNIST
      ↓
dataset.py
      ↓
数据预处理
      ↓
model.py
      ↓
CNN
      ↓
train.py
      ↓
模型训练
      ↓
best_model.pth
      ↓
test.py
      ↓
Test Accuracy
      ↓
Confusion Matrix
```
