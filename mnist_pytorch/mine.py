#!/usr/bin/env python
# coding: utf-8

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import csv
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import pycuda.driver as cuda

import matplotlib 
matplotlib.use("agg")
import matplotlib.pyplot as plt
from models import CNN, DNN

use_cuda = torch.cuda.is_available()
print("Setting Info")
print("============")
print("- use_cuda: ", use_cuda)
print("- Path: ", os.getcwd())
print("- PyTorch", torch.__version__)
print("- Python: ", sys.version)

# GPU settings
if torch.cuda.is_available(): # GPUが利用可能か確認
    device = 'cuda'
else:
    device = 'cpu'

# model reload
model = DNN()
PRETRAINED_MODEL_PATH = "/home/uchiumi/JNNS2019/mnist_pytorch/train_log/dnn_mnist__2019-0425-1923.pth"
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

# load data
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)

X_train_0 = np.asarray(mnist_trainset[0][0]) # image
y_train_0 = mnist_trainset[0][1] # label

# train
train_loader_for_MINE = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=False)

# test
test_loader_for_MINE = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=False)


### Get layer values (the state of each nodes)
def get_nodes_with_train_data(model):
    model.eval()
    list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader_for_MINE):
            result = model(data)
            list.append(result)
    return list

def get_nodes_with_test_data(model):
    model.eval()
    list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader_for_MINE):            
            result = model(data)
            list.append(result)
    return list


list = get_nodes_with_train_data(model)
print(len(list))
print(list[0].keys())
print("model_input:  ", list[0]["model_input"].shape)
print("fc1_output:   ", list[0]["fc1_output"].shape)
print("fc2_output:   ", list[0]["fc2_output"].shape)
print("fc3_output:   ", list[0]["fc3_output"].shape)
print("fc4_output:   ", list[0]["fc4_output"].shape)
print("model_output: ", list[0]["model_output"].shape)

model_input = []
fc1_output = []
fc2_output = []
fc3_output = []
fc4_output = []
model_output = []

for i in range(len(train_loader_for_MINE)):
    model_input.append(list[i]["model_input"].data.numpy().flatten())
    fc1_output.append(list[i]["fc1_output"].data.numpy().flatten())
    fc2_output.append(list[i]["fc2_output"].data.numpy().flatten())
    fc3_output.append(list[i]["fc3_output"].data.numpy().flatten())
    fc4_output.append(list[i]["fc4_output"].data.numpy().flatten())
    model_output.append(list[i]["model_output"].data.numpy().flatten())

model_input = np.array(model_input)
fc1_output = np.array(fc1_output)
fc2_output = np.array(fc2_output)
fc3_output = np.array(fc3_output)
fc4_output = np.array(fc4_output)
model_output = np.array(model_output)


# MINE: Murual Information Neural Estimater
x = model_input
y = fc1_output
z = np.concatenate([x, y], axis=1)

print("x: ", x.shape)
print("y: ", y.shape)
print("z: ", z.shape)

def sample_batch(x, y, sample_size, batch_size=int(1e2), sample_mode='joint'):
    if sample_mode == 'joint':
        index_1 = np.random.choice(range(sample_size), size=batch_size, replace=False)
        z = np.concatenate([x, y], axis=1)
        batch = z[index_1]
    elif sample_mode == 'marginal':
        index_1 = np.random.choice(range(sample_size), size=batch_size, replace=False)
        index_2 = np.random.choice(range(sample_size), size=batch_size, replace=False)
        batch = np.concatenate([x[index_1], y[index_2]], axis=1)
    return batch

# joint & marginal sample
joint_data = sample_batch(x, y, x.shape[0], batch_size=1000, sample_mode='joint')
marginal_data = sample_batch(x, y, x.shape[0], batch_size=1000,sample_mode='marginal')
print(joint_data.shape)
print(marginal_data.shape)

# StatisticsNet
class StatisticsNet(nn.Module):
    def __init__(self, xdim=1, ydim=1, H=512):
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.fc1 = nn.Linear(self.xdim+self.ydim, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, 1)
        
        # guassian noise
        self.std1 = 0.3
        self.std2 = 0.5
        self.std3 = 0.5
        
        self.bn1 = nn.BatchNorm1d(H)
        self.bn2 = nn.BatchNorm1d(H)
        self.bn3 = nn.BatchNorm1d(1)
        
    def gaussian_noise(self, x, stddev):
        return x + torch.autograd.Variable(torch.randn(x.size()).cuda() * stddev)
        
    def forward(self, z_input):
        
        if self.training: z_input = self.gaussian_noise(z_input, self.std1)
        h1 = F.elu(self.fc1(z_input))
        
        if self.training: h1 = self.gaussian_noise(h1, self.std2)
        h2 = F.elu(self.fc2(h1))
        
        if self.training: h2 = self.gaussian_noise(h2, self.std3)
        h3 = self.fc3(h2)
        
        return h3
    
        """
        h1 = self.bn1(F.elu(self.fc1(z_input)))
        h2 = self.bn2(F.elu(self.fc2(h1)))
        h3 = self.fc3(h2)
        """
        

        return x 
        
        # h1 = self.bn1(F.elu(self.fc1(z_input)))
        # h1 = F.dropout(h1, p=0.2, training=True)
        # h2 = F.dropout(h2, p=0.2, training=True)
            
# moving average
def moving_average(array, window_size=100):
    return [np.mean(array[i : i + window_size]) for i in range(0, len(array) - window_size)]

def calc_MI_LowerBound(joint, marginal, snet):
    t = snet(joint)
    et = torch.exp(snet(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et)) # Lower bound for MI
    return mi_lb, t, et

def update(joint_batch, marginal_batch, snet, optimizer, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint_batch = torch.autograd.Variable(torch.FloatTensor(joint_batch)).cuda()
    marginal_batch = torch.autograd.Variable(torch.FloatTensor(marginal_batch)).cuda()
    
    mi_lb , t, et = calc_MI_LowerBound(joint_batch, marginal_batch, snet)
    
    # unbiasing use moving average
        # ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
        # loss = -1 * (torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et)) # original loss function
    # use biased estimator
    loss = - mi_lb
    
    optimizer.zero_grad()
    autograd.backward(loss)
    optimizer.step()    
    return mi_lb, ma_et, loss

def train(x, y, snet, optimizer, csv_file_path, batch_size=100, nb_epoch=int(5e+4), log_freq=int(5e+2)):
    with open(csv_file_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["MI LowerBound", "Loss"])

    mi_lbs = [] # list of MI lower bounds
    ma_mi_lbs = [] # list of moving ave. of mi_lbs
    losses = [] # lsit of loss function
    
    # moving average of exp(T)
    ma_et = 1.
    
    for i in range(nb_epoch):
        if x.shape[0] != y.shape[0]: 
            print("shape error.")
            break
            
        sample_size = x.shape[0]
        joint_batch = sample_batch(x, y, sample_size, batch_size=batch_size, sample_mode='joint')
        marginal_batch = sample_batch(x, y, sample_size, batch_size=batch_size, sample_mode='marginal')
        
        mi_lb, ma_et, loss = update(joint_batch, marginal_batch, snet, optimizer, ma_et)
        mi_lbs.append(mi_lb.detach().cpu().numpy())
        ma_mi_lbs = moving_average(mi_lbs, window_size=50)
        losses.append(loss)
        
        
        # csvファイルに書き込み
        with open(csv_file_path, 'a') as f:
            writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
            writer.writerow([mi_lb.item(), loss.item()])     # list（1次元配列）の場合
        
        if (i + 1) % (log_freq) == 0:
            print('Epoch: {:>6} \t MI lower bounds: {:2.4f} \t Loss of MINE: {:2.4f} \t MI lower bounds (m.a.): {:2.4f}'.format(i+1, mi_lbs[-1], losses[-1], ma_mi_lbs[-1]))
            
    return mi_lower_bounds, losses


model_name = "dnn_mnist__2019-0425-1923"
snet = StatisticsNet(xdim=x.shape[1], ydim=y.shape[1], H=1000).cuda()
# optimizer = optim.Adam(snet.parameters(), lr=1e-3, weight_decay=1e-4)
# optimizer = optim.Adam(snet.parameters(), lr=1e-4)
optimizer = optim.SGD(snet.parameters(), lr=1e-3, momentum=0.8, weight_decay=1e-3)

from datetime import datetime
date_string = datetime.now().strftime("%Y-%m%d-%H%M")
csv_file_name = model_name + "__" + date_string + ".csv"
csv_file_path = os.path.join(os.getcwd(), "mine_log", csv_file_name)
mi_lower_bounds, losses = train(x, y, snet, optimizer, csv_file_path, batch_size=100, nb_epoch=50000, log_freq=100)

