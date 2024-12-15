import os
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim


# 数据加载
class CifarDataset(Dataset):
    def __init__(self,folder, data_type, transform=None, test=False):
        self.folder = folder # "Dataset"
        self.data_type = data_type # "test/train/valid"
        self.txt_file = os.path.join(self.folder, self.data_type + 'set.txt') # 打开相应的txt文件
        self.images_folder = os.path.join(self.folder, 'image') # image文件夹
        self.transform = transform
        self.test = test

        with open(self.txt_file, 'r') as f:
            self.image_label = f.read().splitlines()




    def __getitem__(self, index):
        if self.test:
            image_path = os.path.join(self.images_folder, self.image_label[index])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        else:
            image_name, label = self.image_label[index].split(' ')
            label = int(label)
            image_path = os.path.join(self.images_folder, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            # print(image_name, label)
            return image, label


    def __len__(self):
        # 添加len函数的相关内容
        return len(self.image_label)


# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义模型的网络结构
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size= 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size= 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size= 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 定义模型前向传播的内容
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet = models.resnet18(weights = 'DEFAULT')

        # 修改输入层的通道数，CIFAR-10 图像为 32x32x3，需要将输入通道数从 3 改为 64
        # 因为默认的 ResNet 输入尺寸是 224x224，所以需要调整第一个卷积层（conv1）的输入通道数
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # ResNet 在最后一层的输出是 512 个特征，我们需要将其映射到 CIFAR-10 的 10 类
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)


# 定义 train 函数
def train():
    # 参数设置
    sum_true = 0
    sum_loss = 0.0
    total = 0

    net.train()
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            images, labels_true = data[0].to(device), data[1].to(device)

            # Forward
            labels_hat = net(images)
            loss = loss_func(labels_hat, labels_true)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update
            _, labels_hat = torch.max(labels_hat, 1)
            total += labels_true.size(0)
            sum_true += torch.sum(labels_hat == labels_true).item()
            sum_loss += loss.item()
        train_acc = sum_true / total
        train_loss = sum_loss / (total / batch_size)
        print(
            f"epoch: {epoch+1}, train loss: {train_loss:.4f}, train accuracy: {train_acc * 100:.2f}%,\
                        time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        # 模型训练n轮之后进行验证
        if (epoch+1) % val_num == 0:
            validation()


# 定义 validation 函数
def validation():
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in dev_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # 在这一部分撰写验证的内容
            _, labels_hat = torch.max(net(images), 1)
            correct += torch.sum(labels_hat == labels).item()
            total += labels.size(0)
    # print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)


# 定义 test 函数
def test():
    # 将预测结果写入result.txt文件中，格式参照实验1
    result = []
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            image = data.to(device)
            labels_hat = net(image)
            _, labels_hat = torch.max(labels_hat, 1)
            result += labels_hat.tolist()

    with open("predict_labels_1120213005.txt", "w") as f:
        for label in result:
            f.write(str(label) + "\n")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 超参数
    batch_size = 32
    lr = 0.001
    epoch_num = 20
    val_num = 5

    # 数据增强
    # --------待补充--------
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    ])
    # ---------------------

    # 构建数据集
    train_set = CifarDataset(folder="Dataset", data_type='train', transform = transform)
    dev_set = CifarDataset(folder="Dataset", data_type='valid', transform = transform)
    test_set = CifarDataset(folder="Dataset", data_type='test', transform = transform, test=True)

    # 构建数据加载器
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    dev_loader = DataLoader(dataset=dev_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    # 初始化模型对象
    net = ResNet()

    # 定义损失函数
    loss_func = CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 模型训练
    train()

    # 对模型进行测试，并生成预测结果
    test()