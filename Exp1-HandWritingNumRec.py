import torch
from torch import device
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import time
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from calculator import calculate_mean_std



class HandWritingNumberRecognize_Dataset(Dataset):
    def __init__(self, folder,test=False):
        # 这里添加数据集的初始化内容
        self.folder=folder
        self.image_folder=os.path.join(self.folder,'images')
        self.label_folder=os.path.join(self.folder,'labels.txt')
        self.image_names=os.listdir(self.image_folder)
        self.test=test
        if self.test:
            self.labels = None
        else:
            with open(self.label_folder, 'r') as f:
                self.labels = f.readlines()
        pass

    def __getitem__(self, index):
        # 这里添加getitem函数的相关内容
        image_path=os.path.join(self.image_folder,self.image_names[index])
        image=Image.open(image_path)
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize(mean, std)(image_tensor)
        if self.test:
            return image_tensor
        else:
            label= int(self.labels[index].strip())
            return image_tensor,label

    def __len__(self):
        # 这里添加len函数的相关内容
        return len(self.image_names)


class HandWritingNumberRecognize_Network(torch.nn.Module):
    def __init__(self):
        super(HandWritingNumberRecognize_Network, self).__init__()
        # 此处添加网络的相关结构，下面的pass不必保留
        self.fc1=nn.Linear(28*28,300)
        self.fc2=nn.Linear(300, 100)
        self.out=nn.Linear(100, 10)



    def forward(self, x):
        # 此处添加模型前馈函数的内容，return函数需自行修改
        x = x.view(x.size(0), 28*28)
        print(x.shape)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.out(x)

        return x


def validation():
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    correct = 0
    total = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in data_loader_val:
            images, true_labels = data
            # 在这一部分撰写验证的内容，下面两行不必保留
            label_hat = model(images)
            label_hat = torch.argmax(label_hat, dim=1)
            total += true_labels.size(0)
            correct += (label_hat == true_labels).sum().item()
        accuracy = (correct/total)

    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print(f"当前模型在验证集上的准确率为：{accuracy:.4f}")


def alltest():
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留
    result = []
    model.eval()
    with torch.no_grad():
        for data in data_loader_test:
            label_hat = model(data)
            # print(label_hat)
            label_hat = torch.argmax(label_hat, dim=1)
            # print(label_hat)
            result += label_hat.tolist()

    with open("prediction.txt","w") as f:
        i=0
        for label in result:
            i+=1
            f.write(str(label)+"\n")
        print(i)



def train(epoch_num):
    # 循环外可以自行添加必要内容
    sum_true = 0
    sum_loss = 0.0
    max_valid_acc = 0
    total = 0
    model.train()
    for index, data in enumerate(data_loader_train, 0):
        images, true_labels = data
        # 该部分添加训练的主要内容
        # 必要的时候可以添加损失函数值的信息，即训练到现在的平均损失或最后一次的损失，下面两行不必保留
        label_hat = model(images)
        loss = loss_function(label_hat, true_labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        label_hat = torch.argmax(label_hat, dim=1)
        sum_true += torch.sum(label_hat == true_labels).item()
        sum_loss += loss.item()
        total += true_labels.size(0)
    train_acc = sum_true/total
    train_loss = sum_loss/(total/batch_size)

    print(
        f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc * 100:.2f}%,\
                time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 超参数
    max_epoch = 1  # 自行设置训练轮数
    num_val = 2  # 经过多少轮进行验证
    batch_size = 64
    lr=0.01
    momentum=0.9

    mean, std = calculate_mean_std("dataset/train/images")
    # 构建数据集，参数和值需自行查阅相关资料补充。
    dataset_train = HandWritingNumberRecognize_Dataset("dataset/train")
    dataset_val = HandWritingNumberRecognize_Dataset("dataset/val")
    dataset_test = HandWritingNumberRecognize_Dataset("dataset/test", True)

    # 构建数据加载器，参数和值需自行完善
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    # 初始化模型对象，可以对其传入相关参数
    model = HandWritingNumberRecognize_Network()
    model.to(device)
    # 损失函数设置
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn中的损失函数进行挑选，并进行参数设置

    # 优化器设置
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # torch.optim中的优化器进行挑选，并进行参数设置

    # 然后开始进行训练
    for epoch in range(max_epoch):
        train(epoch)
        # 在训练数轮之后开始进行验证评估
        # if epoch % num_val == 0:
        #     validation()

    # 自行完善测试函数，并通过该函数生成测试结果
    alltest()
