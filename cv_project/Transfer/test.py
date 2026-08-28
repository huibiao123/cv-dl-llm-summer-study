import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay                #用于绘制混淆矩阵

from dataset import get_dataloaders, class_names
from model import get_transfer_model


# =========================
# 读取Config文件
# =========================
from configs.config import (
    NUM_CLASSES,                                                                    #分类数量
    MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    ERROR_EXAMPLES_PATH                                                                                                           #混淆矩阵图片保存的位置
)


# =========================
# 设备选择
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# 获取测试集
# =========================

_, _, test_loader = get_dataloaders()


# =========================
# 创建模型
# =========================

model = get_transfer_model(
    num_classes=NUM_CLASSES
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.eval()


# =========================
# 模型测试
# =========================

correct = 0

total = 0

all_labels = []
all_predictions = []
error_images = []
error_labels = []
error_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # =========================
        # 保存错误样例
        # =========================

        wrong = predictions != labels

        error_images.extend(
            images[wrong].cpu()
        )

        error_labels.extend(
            labels[wrong].cpu().numpy()
        )

        error_predictions.extend(
            predictions[wrong].cpu().numpy()
        )


# =========================
# 计算准确率
# =========================

test_accuracy = correct / total


print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"Correct: {correct}"
)

print(
    f"Total: {total}"
)


# =========================
# 生成并保存混淆矩阵
# =========================

cm = confusion_matrix(                                                 #生成混淆矩阵
    all_labels,
    all_predictions
)


disp = ConfusionMatrixDisplay(                                         #创建一个混淆矩阵的可视化对象
    confusion_matrix=cm,
    display_labels=class_names
)


fig, ax = plt.subplots(                                                #创建 Matplotlib 画布
    figsize=(10, 10) 
)


disp.plot(                                                             #绘制混淆矩阵
    ax=ax,
    xticks_rotation=45,                                                #为了展示所以斜置45°横坐标标签名称
    values_format="d"                                                  #整数
)


plt.title(
    "Fashion-MNIST Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)


plt.show()
# =========================
# 错误样例可视化
# =========================

num_examples = min(
    25,
    len(error_images)
)

fig, axes = plt.subplots(
    5,
    5,
    figsize=(10, 10)
)

for i in range(num_examples):

    ax = axes[i // 5, i % 5]

    image = error_images[i][0].numpy()

    # 还原 Normalize
    image = image * 0.229 + 0.485

    ax.imshow(
    image,
    cmap="gray"
)

    ax.set_title(
        f"True: {class_names[error_labels[i]]}\n"
        f"Pred: {class_names[error_predictions[i]]}",
        fontsize=9
    )

    ax.axis("off")


# 如果错误样例不足25张
for i in range(num_examples, 25):

    axes[i // 5, i % 5].axis("off")


plt.suptitle(
    "Fashion-MNIST MLP Error Examples"
)

plt.tight_layout()


plt.savefig(
    ERROR_EXAMPLES_PATH,
    dpi=300
)


plt.show()