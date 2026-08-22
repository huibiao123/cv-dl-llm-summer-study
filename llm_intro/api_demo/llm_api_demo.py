import os
import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError(
        "未找到 DEEPSEEK_API_KEY，请检查 .env 文件。"
    )


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
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
            "content": "请用简单的语言解释什么是卷积神经网络。"
        }
    ],
    stream=False
)

print("===== 最小聊天示例 =====")
print(response.choices[0].message.content)


prompt_template = """
请解释下面这个计算机视觉概念：

概念：{concept}

要求：
1. 给出简短定义
2. 解释核心思想
3. 给出一个简单应用案例
4. 使用适合初学者的语言
"""

concept = "ResNet"

user_prompt = prompt_template.format(
    concept=concept
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
            "content": user_prompt
        }
    ],
    stream=False
)

print("\n===== Prompt 模板示例 =====")
print(response.choices[0].message.content)


structured_prompt = """
请分析下面的计算机视觉概念：

概念：{concept}

请严格按照 JSON 格式输出。

JSON 必须包含以下字段：

{{
    "concept": "概念名称",
    "definition": "概念定义",
    "applications": ["应用1", "应用2"],
    "difficulty": "入门/中等/困难"
}}

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
    response_format={
        "type": "json_object"
    },
    stream=False
)

content = response.choices[0].message.content

print("\n===== 结构化输出示例 =====")
print(content)

data = json.loads(content)

print("\n===== Python 解析结果 =====")
print("概念：", data["concept"])
print("定义：", data["definition"])
print("应用：", data["applications"])
print("难度：", data["difficulty"])