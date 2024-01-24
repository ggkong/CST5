import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import random
seed = 11032
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# class Model(nn.Module):
#     def __init__(self, init_weights=True):
#         super(Model, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(1, 8)),
#             nn.ReLU(inplace=True),
#         )
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2)
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 32),
#             nn.ReLU(inplace=True),
#             nn.Linear(32, 1),
#             nn.Sigmoid(),
#         )
#
#
#     def forward(self, x):
#         x = x.view(-1, 128, 8)
#         x = x.unsqueeze(1)
#         x = self.conv(x)
#         x = x.squeeze(3)
#         x = self.encoder_layer(x)
#         x = torch.mean(x, dim=1)
#         x = self.classifier(x)
#         return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 3000)
        self.fc2 = torch.nn.Linear(3000, 1024)
        self.fc3 = torch.nn.Linear(1024, 512)
        self.fc4 = torch.nn.Linear(512, 1)

        self.bn1 = torch.nn.BatchNorm1d(3000)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(512)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        out = self.fc1(x)
        out = torch.nn.Dropout(0.3)(out)
        out = torch.nn.ReLU()(out)
        out = self.bn1(out)

        out = self.fc2(out)
        out = torch.nn.Dropout(0.3)(out)
        out = torch.nn.ReLU()(out)
        out = self.bn2(out)

        out = self.fc3(out)
        out = torch.nn.Dropout(0.3)(out)
        out = torch.nn.ReLU()(out)
        out = self.bn3(out)

        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

class argparse():
    pass

args = argparse()
args.epochs, args.learning_rate, args.train_batch_size, args.test_batch_size = [1000, 0.001, 2048, 2048]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]

class Dataset_CAR(Dataset):
    def __init__(self, flag='train', csv_paths = []):
        assert flag in ['train', 'test'] # flag 必须是train  test 之间的其中一个
        self.flag = flag
        self.__load_data__(csv_paths)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def __load_data__(self, csv_paths: list):
        # 读取 排列按照 train feature train label test feature test label
        self.x = torch.tensor(pd.read_csv(csv_paths[0]).values)
        self.y = torch.tensor(pd.read_csv(csv_paths[1], header = None).values) # 因为 label的表头是没有的，所以使用 header  = None
        print("feature shape: {}, label shape: {}".format(self.x.shape, self.y.shape))

csv_path_train = ['K_onlySMOTE_x.csv', 'K_onlySMOTE_y.csv']
train_dataset = Dataset_CAR(flag='train', csv_paths=csv_path_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size = args.train_batch_size, shuffle=True)

csv_path_test = ["K_test_feature.csv", "K_test_y.csv"]
test_dataset = Dataset_CAR(flag='test', csv_paths=csv_path_test)
test_dataloader = DataLoader(dataset=test_dataset, batch_size = args.test_batch_size, shuffle=True)


model = Model().to(args.device)
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # Adam 优化器

train_loss = []
test_loss = []
train_epochs_loss = []
test_epochs_loss = []

from Measurement import compAUC
from Measurement import SN_SP_MCC

for epoch in range(args.epochs):
    model.train()  # 设置模型为训练模式
    for idx, (data, labels) in enumerate(train_dataloader, 0):
        # 前向传播
        data  = data.type(torch.float32).to(args.device)
        outputs = model(data).to(args.device)
        labels = labels.type(torch.float32).to(args.device)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在验证阶段不计算梯度
        correct = 0
        total = 0
        outputs_list = []
        labels_list = []
        pred_list = []
        for idx, (data, labels) in enumerate(test_dataloader, 0):
            data = data.type(torch.float32).to(args.device)
            labels = labels.type(torch.float32).to(args.device)
            outputs = model(data)
            predicted = (outputs > 0.5).float()  # 阈值设为 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            outputs_list.append(outputs)
            labels_list.append(labels)
            pred_list.append(predicted)

        all_outputs = torch.cat(outputs_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        all_pred = torch.cat(pred_list, dim=0)


    SN_SP_MCC(all_labels, all_pred)
    AUC = compAUC(all_labels, all_outputs)
    print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%')
    print(AUC)