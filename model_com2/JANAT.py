import math


from torch.nn import LSTMCell
import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
X = np.load('X.npy')
Y = np.load('Y.npy')
print(X.shape)
print(Y.shape)
X = torch.from_numpy(X).to(torch.float32)
Y = torch.from_numpy(Y).to(torch.float32)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.input_size = 2
        self.hidden_size = hidden_size

        self.linear_ia = nn.Linear(1, hidden_size)
        self.linear_ha = nn.Linear(hidden_size, hidden_size)

        self.linear_ip1 = nn.Linear(1, hidden_size)
        self.linear_hp1 = nn.Linear(hidden_size, hidden_size)

        self.linear_ip2 = nn.Linear(1, hidden_size)
        self.linear_hp2 = nn.Linear(hidden_size, hidden_size)

        self.linear_fu = nn.Linear(hidden_size, hidden_size)
        self.linear_fh = nn.Linear(hidden_size, hidden_size)

        self.linear_gu = nn.Linear(hidden_size, hidden_size)
        self.linear_gh = nn.Linear(hidden_size, hidden_size)

        self.init_parameters()

    def init_parameters(self):
        # 初始化全连接层参数
        for layer in [self.linear_ia, self.linear_ha, self.linear_ip1, self.linear_hp1, self.linear_ip2,
                      self.linear_hp2, self.linear_fu, self.linear_fh, self.linear_gu, self.linear_gh]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, input, hidden):
        x = input[:, 0].unsqueeze(1)
        theta = input[:, 1].unsqueeze(1)
        h_t = hidden
        a_t = torch.tanh(self.linear_ia(x) + self.linear_ha(h_t))
        p1_t = torch.tanh(self.linear_ip1(torch.cos(theta)) + self.linear_hp1(h_t))
        p2_t = torch.tanh(self.linear_ip2(torch.sin(theta)) + self.linear_hp2(h_t))
        un = a_t * p1_t * p2_t * (1 - a_t) * (1 - p1_t) * (1 - p2_t)
        f_t = torch.sigmoid(self.linear_fu(un) + self.linear_fh(h_t))
        g_t = torch.tanh(self.linear_gu(un) + self.linear_gh(h_t))
        h_next = f_t * h_t + (1 - f_t) * g_t
        return h_next


class NET(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NET, self).__init__()
        self.lstm = CustomGRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 2)
        self.hiddensize = hidden_size

    def forward(self, x):
        outputs1 = []
        seq_length = x.shape[1]
        batchsize = x.shape[0]
        hidden_state = torch.zeros(batchsize, self.hiddensize)
        hidden_state=hidden_state.to(device)
        for t in range(seq_length):
            hidden_state = self.lstm(x[:, t], hidden_state)
            outputs1.append(hidden_state)
        outputs = torch.stack(outputs1, dim=1)
        outputs = self.linear(outputs)
        return outputs

'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
# 创建你的模型
model = NET(2, 16)

# 打印模型的参数量
num_params = count_parameters(model)
print(f"Total number of parameters: {num_params}")
import torch
import torchvision.models as models
from thop import profile



input_data = torch.randn(1,60,2)

# 使用 thop 的 `profile` 函数来计算模型的FLOPs
flops, params = profile(model, inputs=(input_data,))
print(f"FLOPs: {flops } FLOPs")
'''
model = NET(2, 16)
model = model.to(device)
model.load_state_dict(torch.load('JANET.pt'))
criterion = nn.MSELoss()
criterion = criterion
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)
n_epochs = 100
from torch.utils.data import TensorDataset, DataLoader
train_loader = DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=True)
for epoch in range(n_epochs):
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x=x.to(device)
        y_pred = model(x)
        y=y.to(device)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            scheduler.step(loss.item())
            torch.save(model.state_dict(), 'JANET.pt')
            print(f'Epoch: {epoch + 1}/{n_epochs}, Step: {i}/{len(train_loader)}, Loss: {loss.item():.8f}')



