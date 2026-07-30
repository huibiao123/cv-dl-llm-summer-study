张量、数据集、训练集/验证集/测试集、线性回归、softmax、交叉熵、梯度下降、反向传播。
# 张量（Tensor）

定义 ：张量就是机器学习中表示数据的数学结构。

简要理解：

| 数学概念       | 机器学习对应 |
| ---------- | ------ |
| 标量（Scalar） | 一个数字   |
| 向量（Vector） | 一维数组   |
| 矩阵（Matrix） | 二维数组   |
| 张量（Tensor） | 多维数组   |
# 数据集（Dataset）

定义：用来训练机器学习模型的数据集合。

# 训练集 / 验证集 / 测试集

## 训练集

定义：让模型用于学习的数据集。

## 验证集

定义：用于选择和调整模型的数据集。

## 测试集

定义：用于最终测试模型的数据集。

# 线性回归（Linear Regression）

定义：计算机自己找到y=wx+b中的最近似的w和b
存在误差叫：残差（Residual）

# Softmax

定义： Softmax 把一组任意实数转换成 0~1 之间，并且总和为1的概率分布。

# 交叉熵（Cross Entropy）

定义：衡量模型预测和真实答案差距的方法。

理解：用于衡量模型错得有多严重。

# 梯度下降（Gradient Descent）

定义：负责调整模型参数，找到让错误最小的参数。

# 反向传播（Backpropagation）

理解：负责计算“参数应该如何调整”。

| 前向传播 | 反向传播 | 参数更新 |
|---|---|---|
| 输入<br>↓<br>神经网络<br>↓<br>预测结果<br>↓<br>Loss | Loss<br>↓<br>计算梯度<br>↓<br>参数影响程度 | 梯度下降<br>↓<br>修改权重 |
# 案例讲解

给模型一张图片，让它判断：这是猫还是狗。

|图片|耳朵大小(x₁)|脸长度(x₂)|标签|
|---|---|---|---|
|图片A|8|5|猫|
|图片B|2|9|狗|
|图片C|7|6|猫|
|图片D|3|8|狗|
标签为 猫=0  狗=1
## 张量 Tensor
图片A-> x = [ 8,5 ]

## 线性回归模型

假设一个y对应于标签y=w1​x1​+w2​x2​+b
那么此时对于每一个标签都存在一个标签y=w1​x1​+w2​x2​+b（猫狗共有两套）

# Softmax

例如假设此时的两个值y1(猫)=2.5 y2(狗)=1.0
Softmax转换后为：
猫概率:0.82
狗概率:0.18
即：82% 是猫

# 交叉熵

[0.82,0.18]对比与[1,0]
Loss=0.19
表示模型的错误小

假如是[0.1.0.9]对比与[1,0]
Loss=2.3
表示模型的错误大

# 反向传播 Backpropagation

Loss
↓
计算每个参数责任
↓
得到梯度

决定是w1、w2的调整方向

# 梯度下降 Gradient Descent

得到调整方向以后：w(new)=w−η∇L
- w：参数
- η：学习率
- ∇L：梯度

# 实验记录
```
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)                      #设置 PyTorch 的随机数种子
x = torch.linspace(0,10,100)              #从0到10随机生成100个数据
y = 3*x + 2                               #让我们假定实际的线性关系是Y=3X+2
y += torch.randn(100)                     #加入100个（等同于数据量）随机的生态分布噪声

model = torch.nn.Linear(                  #创建一个线性回归模型
    1,                                    #X的数量（因为可能是Y=aX1+bX2+C等等)
    1                                     #Y的数量（参考↑）
)

  

loss_fn = torch.nn.MSELoss()              #创建损失函数

learning_rate = 0.001                      #学习率

optimizer = torch.optim.SGD(              #随机梯度下降优化器负责根据梯度修改参数
    model.parameters(),                   #把模型里面所有需要训练的参数交给优化器
    lr=learning_rate                      #每次调整幅度
)

epochs = 100                              #训练轮数
batch_size = 10                           #单次训练使用的样本数
loss_list=[]                              #保存每一次训练产生的loss

for epoch in range(epochs):               #循环训练
    total_loss=0                          #每一轮循环开始重置loss
    for i in range(0,len(x),batch_size):  #range用于切分单次训练使用的样本

        x_batch=x[i:i+batch_size]         #取出X的值
        y_batch=y[i:i+batch_size]         #取出Y的值

        x_batch=x_batch.reshape(-1,1)     #reshape以符合上面的Linear的输入要求，-1表示让计算机自己计算，而后面的1表示一个输入变量
        y_batch=y_batch.reshape(-1,1)     #同上，但是后面的1指一个输出

        pred=model(x_batch)               #计算预测值

        loss=loss_fn(                     #计算loss值
            pred,
            y_batch
        )

        optimizer.zero_grad()             #清空旧梯度
        loss.backward()                   #反向传播
        optimizer.step()                  #更新参数 参数（new）=参数−学习率×梯度
        total_loss += loss.item()         #累计loss

    avg_loss=total_loss/(len(x)/batch_size) #计算平均loss
    loss_list.append(avg_loss)            #保存loss，用于绘制训练曲线

    if epoch%10==0:                       #每10轮打印一次
        print(
            "epoch:",
            epoch,
            "loss:",
            avg_loss
        )
 
print(                                    #输出最后的结果
    "学习后的参数:"
)
print(                                    #查看y=wx+b的w和b值
    model.weight,                         #模型的w
    model.bias                            #模型的b
)
plt.plot(loss_list)                       #绘制loss变化曲线
plt.xlabel("epoch")                       #横坐标epoch
plt.ylabel("loss")                        #纵坐标loss
plt.title(
    f"lr={learning_rate}, batch={batch_size}" #标题
)
plt.show()                                #显示图片
```
结果：
![](assets/Pasted%20image%2020260730121537.png)
![](assets/Pasted%20image%2020260730121606.png)

变更lr=0.0001后：
![](assets/Pasted%20image%2020260730121815.png)

变更lr=0.00001后：
![](assets/Pasted%20image%2020260730121853.png)

可以发现lr的后两张图近乎等于将第一张图的横坐标分别膨胀10/100倍
（这是因为这次的图的对于线性关系的实际结果（3和2）来说，0.01，0.001，0.0001的结果较为有限，所以呈现这种结果

变更lr=0.1后：
![](assets/Pasted%20image%2020260730122329.png)

非常明显的出现了梯度爆炸的情况

变更batch size=1后：
![](assets/Pasted%20image%2020260730122706.png)

变更batch size=100后：
![](assets/Pasted%20image%2020260730122747.png)

可以发现
当batch size减小到1时，每个样本都会触发一次参数更新，下降速度更快
而当batch size增大到100时，每次参数更新基于完整训练集计算梯度，因此loss下降过程更加平滑，但每轮训练参数更新次数减少，收敛速度相对较慢。

## 训练流程图
```
        输入数据
           |
           v
      x,y训练样本
           |
           v

     ----------------
     |   神经网络    |
     |   y=w*x+b     |
     ----------------

           |
           v

       预测结果 pred

           |
           v

     计算 Loss

     Loss=(pred-y)^2

           |
           v

      loss.backward()

           |
           v

       计算梯度

          dw
          db

           |
           v

     optimizer.step()

           |
           v

       更新参数

       w = w - dw
       b = b - db

           |
           v

        下一轮训练
```
## 一次训练迭代
```text
输入一批数据(x_batch, y_batch)
            ↓
      模型预测
      pred=model(x_batch)
            ↓
      计算误差
      loss=loss_fn(pred,y_batch)
            ↓
      计算参数应该怎么调整
      loss.backward()
            ↓
      更新模型参数
      optimizer.step()
```