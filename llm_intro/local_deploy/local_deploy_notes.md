
# Local Deployment Notes

## 1. 实验目的
使用本地部署工具运行大语言模型，完成本地模型推理测试。
本次实验选择 Ollama 作为本地推理工具，并使用 DeepSeek-R1 7B 模型进行测试。
## 2. 环境信息
### 操作系统
Windows
### 推理工具
Ollama
### Python 环境
Python 3.11.9
### Ollama 版本
使用以下命令查看 Ollama 版本：
```
ollama --version
````
实际版本：
![](`screenshots/ollama_version.png`)
## 3. 模型信息

### 模型名称
DeepSeek-R1 7B
### 模型运行方式
使用 Ollama 在本地运行，不通过远程 API 调用。
### 下载模型
```
ollama pull deepseek-r1:7b
```
### 查看已安装模型
```
ollama list
```
## 4. 启动命令
模型下载完成后，使用以下命令启动本地推理：
```
ollama run deepseek-r1:7b
```
启动后可以直接在终端输入问题，并由本地模型生成回答。
## 5. 推理测试
### 测试问题
```
请解释什么是卷积神经网络，并说明卷积层的作用。
```
### 推理命令
```
ollama run deepseek-r1:7b
```
启动模型后输入测试问题，模型在本地完成推理并返回结果。
### 推理结果
模型能够正常返回关于卷积神经网络的解释，说明本地模型部署和推理流程正常。
推理截图：
![](`screenshots/inference_result.png`)

## 6. 实验结果

本次实验成功完成了基于 Ollama 的本地大语言模型部署和推理。

实验主要完成以下内容：

1. 安装并配置 Ollama。
2. 检查 Ollama 是否能够正常运行。
3. 下载 DeepSeek-R1 7B 模型。
4. 使用 `ollama run` 启动本地模型。
5. 输入计算机视觉相关问题。
6. 获取本地模型生成的回答。
本次实验使用本地部署的模型完成推理，不需要通过远程 LLM API 获取回答。
## 7. 实验截图
### Ollama 环境
![](`screenshots/inference_result.png`)
用于记录 Ollama 安装和版本信息。
### 本地模型
![](`screenshots/inference_result.png`)
用于记录 DeepSeek-R1 7B 模型已经下载到本地。
### 本地推理
![](`screenshots/inference_result.png`)

用于记录 DeepSeek-R1 7B 实际运行并完成问题回答的过程。