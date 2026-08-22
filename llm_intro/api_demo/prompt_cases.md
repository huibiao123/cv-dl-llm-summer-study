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

| 类型              | 示例                   | 主要作用                   |
| --------------- | -------------------- | ---------------------- |
| 基础 Prompt       | 解释 CNN               | 测试最基本的 API 对话          |
| Prompt Template | `{concept}` + ResNet | 让同一个 Prompt 可以重复用于不同概念 |
| 结构化 Prompt      | JSON 格式的语义分割分析       | 让模型输出能够被程序继续处理         |

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
