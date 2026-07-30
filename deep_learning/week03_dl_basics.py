import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
x = torch.linspace(0,10,100)
y = 3*x + 2
y += torch.randn(100)

model = torch.nn.Linear(
    1,
    1
)

loss_fn = torch.nn.MSELoss()

learning_rate = 0.001

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

epochs = 100
batch_size = 100
loss_list=[]

for epoch in range(epochs):

    total_loss=0


    for i in range(0,len(x),batch_size):

        x_batch=x[i:i+batch_size]
        y_batch=y[i:i+batch_size]

        x_batch=x_batch.reshape(-1,1)
        y_batch=y_batch.reshape(-1,1)


        pred=model(x_batch)

        loss=loss_fn(
            pred,
            y_batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss=total_loss/(len(x)/batch_size)

    loss_list.append(avg_loss)


    if epoch%10==0:
        print(
            "epoch:",
            epoch,
            "loss:",
            avg_loss
        )

print(
    "学习后的参数:"
)
print(
    model.weight,
    model.bias
)

plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title(
    f"lr={learning_rate}, batch={batch_size}"
)
plt.show()