import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import torch
import torchvision.models as models
from scipy import io
from thop import profile
device=torch.device('cpu')
input=np.random.rand(1,7,30)
output=np.random.rand(1,2)
'''
complex_numbers = input[:,:,0] + 1j * np.array(input[:,:,1])
magnitude = np.abs(complex_numbers)
phase = np.angle(complex_numbers)
magnitude=torch.from_numpy(magnitude).to(torch.float32)
phase=torch.from_numpy(phase).to(torch.float32)
context_input=torch.concatenate((magnitude.unsqueeze(2),phase.unsqueeze(2)),dim=-1)
print(context_input.shape)#torch.Size([1, 12, 2])
'''

class VDLSTM(nn.Module):
    def __init__(self, M,G):
        super(VDLSTM, self).__init__()
        self.M = M
        self.G=G
        self.LSTM=nn.LSTM(input_size=15,hidden_size=G,num_layers=1,batch_first=True)
        self.G_wI1 = [nn.Linear(G, 1, bias=False) for _ in range(self.M)]
        self.G_wI2 = [nn.Linear(G, 1, bias=False) for _ in range(self.M)]
        self.G_wQ1 = [nn.Linear(G, 1, bias=False) for _ in range(self.M)]
        self.G_wQ2 = [nn.Linear(G, 1, bias=False) for _ in range(self.M)]
        self.h_g = nn.Tanh()
        self.OI = nn.Linear(2 * M, 1, bias=False)
        self.OQ = nn.Linear(2 * M, 1, bias=False)

    def forward(self, x):
        x_l = x[:, :, :15]
        x_theta = x[:, -1, 15:] #torch.Size([1, 15]) theta
        cos_theta=torch.cos(x_theta)
        sin_theta=torch.sin(x_theta)#(1,15)
        print(x_theta.shape,'theta')
        print(x_l.shape,'l')
        A,_ = self.LSTM(x_l)
        A=A[:, -1, :]
        print(A.shape)
        I_out = 0
        Q_out = 0
        for i in range(15):
            temI1=self.G_wI1[i](A)*cos_theta[:, i]
            temI2=self.G_wI2[i](A)*sin_theta[:, i]
            I_out=I_out+temI1+temI2
            temQ1=self.G_wQ1[i](A)*cos_theta[:, i]
            temQ2=self.G_wQ2[i](A)*sin_theta[:, i]
            Q_out=Q_out+temQ1+temQ2

        output = torch.cat((I_out, Q_out), dim=1)

        return output

model = VDLSTM(M=16,G=16)
context_input = torch.from_numpy(input).to(torch.float32)
out = model(context_input)
print(out.shape,'out')
input_data = context_input
# 使用 thop 的 `profile` 函数来计算模型的FLOPs
flops, params = profile(model, inputs=(input_data,))
print(f"FLOPs: {flops} FLOPs")
print(f"Params: {params} Params")
X = np.load('X.npy')


'''
complex_numbers = X[:, :, 0] + 1j * np.array(X[:, :, 1])
magnitude = np.abs(complex_numbers)
phase = np.angle(complex_numbers)
magnitude = torch.from_numpy(magnitude).to(torch.float32)
phase = torch.from_numpy(phase).to(torch.float32)
X = torch.concatenate((magnitude.unsqueeze(2), phase.unsqueeze(2)), dim=-1)

Y = np.load('Y.npy')
print(X.shape)
print(Y.shape)
X = torch.from_numpy(X).to(torch.float32)
Y = torch.from_numpy(Y).to(torch.float32)
model = model.to(device)
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
        x = x.to(device)
        y_pred = model(x)
        y = y.to(device)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            scheduler.step(loss.item())
            torch.save(model.state_dict(), 'VDLSTM.pt')
            print(f'Epoch: {epoch + 1}/{n_epochs}, Step: {i}/{len(train_loader)}, Loss: {loss.item():.8f}')

X_test = np.load('X.npy')

complex_numbers = X_test[:, :, 0] + 1j * np.array(X_test[:, :, 1])
magnitude = np.abs(complex_numbers)
phase = np.angle(complex_numbers)
magnitude = torch.from_numpy(magnitude).to(torch.float32)
phase = torch.from_numpy(phase).to(torch.float32)
X_test = torch.concatenate((magnitude.unsqueeze(2), phase.unsqueeze(2)), dim=-1)

Y_test = np.load('Y.npy')
print(X_test.shape)
print(Y_test.shape)
X_test = torch.from_numpy(X_test).to(torch.float32)
Y_test = torch.from_numpy(Y_test).to(torch.float32)
model.load_state_dict(torch.load('VDLSTM.pt'))
Y_gg = []
batch_size = 20
for i in range(0, X_test.shape[0], batch_size):
    if i + batch_size > X_test.shape[0]:
        batch_size = X_test.shape[0] - i
    x_test = X_test[i:i + batch_size]
    Y_pred_no_inverse_scaler = model(x_test)
    Y_pred = Y_pred_no_inverse_scaler  # 1,60,2
    print(Y_pred.shape)
    Y_I = Y_pred[:, 0].detach().numpy()
    Y_Q = Y_pred[:, 1].detach().numpy()
    Y_I = Y_I.flatten()
    Y_Q = Y_Q.flatten()
    Y = Y_I + 1j * Y_Q
    Y_gg+=Y_I.tolist()

Y = np.array(Y_gg)
print(Y)
path = 'K18m_PA_Data.mat'
mat = io.loadmat(path)
real_out = np.array(mat['iq_out'])
real_out = real_out.reshape(len(real_out))
def nmse(target, estimate):
    mse = np.mean((target - estimate) ** 2)
    target_energy = np.mean(target ** 2)
    nmse_value = mse / target_energy
    nmse_value = 10 * np.log10(nmse_value)
    return nmse_value

print(nmse(Y, np.real(real_out)[0:len(Y)]), "NMSE")
'''


