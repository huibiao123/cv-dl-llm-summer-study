确定最终视觉项目题目、数据集、类别、评价指标；完成数据集说明和初步预处理。
# 项目题目
## 基于 CNN 的服装图像分类
本项目使用 Fashion-MNIST 数据集，利用卷积神经网络（CNN）对不同类型的服装图像进行分类。
项目后续计划先使用一个简单的 MLP 作为基础模型，再使用 CNN 进行训练，通过测试集准确率和混淆矩阵比较两种模型的分类效果。
# 数据集
## Fashion-MNIST

Fashion-MNIST 是一个用于图像分类的公开数据集，可以看作是 MNIST 手写数字数据集的一个替代数据集。
与 MNIST 不同，Fashion-MNIST 中的图片不是数字，而是各种服装和鞋类。

### 数据集特点：

|项目|内容|
|---|---|
|数据集|Fashion-MNIST|
|训练集|60,000 张|
|测试集|10,000 张|
|图片大小|28 × 28|
|图片类型|灰度图|
|通道数|1|
|类别数量|10 类|
|任务类型|图像分类|

因为图片尺寸比较小、类别数量固定，而且可以直接通过 `torchvision` 加载，所以比较适合用来完成一个完整的 CNN 图像分类实验。
# 类别
|Label|类别|中文说明|
|--:|---|---|
|0|T-shirt/top|T恤/上衣|
|1|Trouser|裤子|
|2|Pullover|套衫|
|3|Dress|连衣裙|
|4|Coat|外套|
|5|Sandal|凉鞋|
|6|Shirt|衬衫|
|7|Sneaker|运动鞋|
|8|Bag|包|
|9|Ankle boot|短靴|
预计可以在程序中分别标注为
```
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
nn.Linear(__, 10)
```
# 评价指标
**Accuracy（准确率）**
$$
Accuracy = \frac{正确预测的样本数量}{全部样本数量}​
$$
此外，为了进一步观察不同服装类别之间的分类情况，可以使用：
**Confusion Matrix（混淆矩阵）**
# 数据集划分
Fashion-MNIST 原始数据是：
训练集：60,000
测试集：10,000
为了在训练过程中观察模型效果，可以进一步从训练集中划出一部分作为验证集。
参考先前做手写数字分类项目，可以分为
Fashion-MNIST
│
├── Train：50,000
├── Validation：10,000
└── Test：10,000
- **Train**：用于训练模型
- **Validation**：用于训练过程中观察模型效果、调整参数
- **Test**：训练完成后进行最终评价
# **初步预处理**
Fashion-MNIST 原始图片是：
28 × 28
灰度图
像素值：0～255
## 第一步：转换为 Tensor
使用： transforms.ToTensor()
将图片转换成 PyTorch Tensor，同时把像素值从：0～255
转换到：0～1
## 第二步：标准化
可以使用：transforms.Normalize((0.5,), (0.5,))
将数据进一步进行标准化处理。
## 总结
```
原始图片
  ↓
28 × 28 灰度图
  ↓
ToTensor
  ↓
[1, 28, 28]
  ↓
Normalize
  ↓
输入 CNN
```
# data_loader.py 初稿
```
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


torch.manual_seed(0)


DATA_DIR = "./deep_learning/data"

BATCH_SIZE = 64


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ]
)


full_train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
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


if __name__ == "__main__":
    print(
        "Train:",
        len(train_dataset)
    )

    print(
        "Validation:",
        len(val_dataset)
    )

    print(
        "Test:",
        len(test_dataset)
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
```
# llm_api_demo部分
注，本次使用deepseek进行
```
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
 

load_dotenv()                                            #加载环境变量部分
api_key = os.getenv("DEEPSEEK_API_KEY")                  #使用env中的api

if not api_key:                                          #失败报错
    raise ValueError(
        "未找到 DEEPSEEK_API_KEY，请检查 .env 文件。"
    )
    
client = OpenAI(                                         #创建一个 OpenAI 风格的 API 客户端，但把实际请求发送到DeepSeek的 API 服务        
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(               #最小聊天示例，这里是通过刚才创建好的client，向聊天模型发送一次请求，然后将结果返回到response
    model="deepseek-v4-flash",                           #选择模型 
    messages=[                                           #设定对话消息的背景与内容
        {
            "role": "system",                            #对系统/模型发出指令
            "content": "你现在负责帮助我学习图像处理的内容。"
        },
        {
            "role": "user",                              #用户的身份提出的问题
            "content": "请用简单的语言解释什么是卷积神经网络。"
        }
    ],
    stream=False                                         #等模型把完整回答生成完，再一次性返回给Python
)

print("===== 最小聊天示例 =====")
print(response.choices[0].message.content)               #输出response的内容


prompt_template = """                                    # Prompt 模板示例
请解释下面这个计算机视觉概念：

概念：{concept}                                           #留一个作为占位

要求：

1. 给出简短定义

2. 解释核心思想

3. 给出一个简单应用案例

4. 使用适合初学者的语言

"""
concept = "ResNet"                                       #设定占位是ResNet
user_prompt = prompt_template.format(                    #把模板里的{concept}替换成变量concept的实际内容
    concept=concept
)
response = client.chat.completions.create(               #这里是通过刚才创建好的client，向聊天模型发送一次请求，然后将结果返回到response
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "system",
            "content": "你现在负责帮助我学习图像处理的内容。"
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    stream=False
)

print("\n===== Prompt 模板示例 =====")
print(response.choices[0].message.content)

structured_prompt = """                                    #结构化输出示例

请分析下面的计算机视觉概念：

  

概念：{concept}

  

请严格按照 JSON 格式输出。

  

JSON 必须包含以下字段：

  

{{                                                         #重复以真正的使用{

    "concept": "概念名称",

    "definition": "概念定义",

    "applications": ["应用1", "应用2"],

    "difficulty": "入门/中等/困难"

}}                                                          #重复以真正的使用}

  

不要输出 JSON 以外的内容。

"""
structured_prompt = structured_prompt.format(
    concept="语义分割"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {
            "role": "system",
            "content": "你现在负责帮助我学习图像处理的内容。"
        },
        {
            "role": "user",
            "content": structured_prompt
        }

    ],
 
    response_format={                                         #这次请求要求返回JSON对象
        "type": "json_object"
    },
    stream=False

)
content = response.choices[0].message.content                 #输出
print("\n===== 结构化输出示例 =====")
print(content)

data = json.loads(content)                                    # 将 JSON 字符串转换为 Python可以输出的内容，然后在下面分条目输出
print("\n===== Python 解析结果 =====")
print("概念：", data["concept"])
print("定义：", data["definition"])
print("应用：", data["applications"])
print("难度：", data["difficulty"])
)
```
## prompt_cases.md
````
# Prompt Cases

## 1. 最小聊天示例

### 使用场景
使用 DeepSeek API 对计算机视觉和深度学习中的基础概念进行简单解释。

### System Prompt
```text
你现在负责帮助我学习图像处理的内容。
```

### User Prompt
```text
请用简单的语言解释什么是卷积神经网络。
```

### 设计目的
通过一个比较简单的问题测试 DeepSeek API 是否能够正常返回文本，并观察模型对基础计算机视觉概念的解释效果。

---

## 2. Prompt 模板示例

### 使用场景
针对不同的计算机视觉概念，使用相同的 Prompt 模板生成学习内容。

### System Prompt
```text
你现在负责帮助我学习图像处理的内容。
```

### Prompt Template
```text
请解释下面这个计算机视觉概念：

概念：{concept}

要求：
1. 给出简短定义
2. 解释核心思想
3. 给出一个简单应用案例
4. 使用适合初学者的语言
```

### 当前测试输入
```text
ResNet
```

### 实际生成的 Prompt
```text
请解释下面这个计算机视觉概念：

概念：ResNet

要求：
1. 给出简短定义
2. 解释核心思想
3. 给出一个简单应用案例
4. 使用适合初学者的语言
```

### 设计目的
将具体的概念名称设置为变量 `{concept}`，从而使同一个 Prompt 可以用于不同的计算机视觉概念。

例如可以将：

```text
ResNet
```

替换为：

```text
CNN
```

或者：

```text
Vision Transformer
```

这样不需要重新设计整个 Prompt。

---

## 3. 结构化输出示例

### 使用场景
让 DeepSeek 对计算机视觉概念进行分析，并按照指定的 JSON 格式返回结果，使模型输出能够进一步被 Python 程序处理。

### System Prompt
```text
你现在负责帮助我学习图像处理的内容。
```

### Prompt Template
```text
请分析下面的计算机视觉概念：

概念：{concept}

请严格按照 JSON 格式输出。

JSON 必须包含以下字段：

{
    "concept": "概念名称",
    "definition": "概念定义",
    "applications": ["应用1", "应用2"],
    "difficulty": "入门/中等/困难"
}

不要输出 JSON 以外的内容。
```

### 当前测试输入
```text
语义分割
```

### 实际生成的 Prompt
```text
请分析下面的计算机视觉概念：

概念：语义分割

请严格按照 JSON 格式输出。

JSON 必须包含以下字段：

{
    "concept": "概念名称",
    "definition": "概念定义",
    "applications": ["应用1", "应用2"],
    "difficulty": "入门/中等/困难"
}

不要输出 JSON 以外的内容。
```

### 输出格式
程序通过 API 的 `response_format` 指定：

```python
response_format={
    "type": "json_object"
}
```

要求模型返回 JSON 格式的数据。

预期的数据结构为：

```json
{
    "concept": "概念名称",
    "definition": "概念定义",
    "applications": [
        "应用1",
        "应用2"
    ],
    "difficulty": "入门/中等/困难"
}
```

### Python 解析
模型返回的 JSON 内容首先以字符串形式保存：

```python
content = response.choices[0].message.content
```

然后使用：

```python
data = json.loads(content)
```

将 JSON 字符串转换为 Python 字典。

之后可以通过字段名称获取具体内容：

```python
data["concept"]
data["definition"]
data["applications"]
data["difficulty"]
```

这样模型生成的内容就可以被 Python 程序进一步处理，而不仅仅是作为普通文本显示。

---

## 4. Prompt 设计总结

本次实验主要使用了三种 Prompt 使用方式：

| 类型 | 示例 | 主要作用 |
| --- | --- | --- |
| 基础 Prompt | 解释 CNN | 测试最基本的 API 对话 |
| Prompt Template | `{concept}` + ResNet | 让同一个 Prompt 可以重复用于不同概念 |
| 结构化 Prompt | JSON 格式的语义分割分析 | 让模型输出能够被程序继续处理 |

整体调用流程为：

```text
用户输入
    ↓
Prompt
    ↓
DeepSeek API
    ↓
模型生成结果
    ↓
普通文本 / JSON
    ↓
Python 程序处理
```

通过这几个案例，可以实现从简单的 LLM 对话，到 Prompt 模板，再到结构化输出的基本 API 调用流程。
````
## .env.example
这里是为了展示如何使用.env来调用api并且避免直接泄露api
```
DEEPSEEK_API_KEY=your_api_key_here
```
## 终端部分的输出
![](assets/Pasted%20image%2020260822130611.png)
![](assets/Pasted%20image%2020260822130627.png)
![](assets/Pasted%20image%2020260822130638.png)