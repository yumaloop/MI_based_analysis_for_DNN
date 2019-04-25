#!/usr/bin/env python
# coding: utf-8

# MINE paper: https://arxiv.org/pdf/1801.04062.pdf

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import pycuda.driver as cuda
from datetime import datetime

use_cuda = torch.cuda.is_available()
print("Setting Info")
print("=========")
print("- use_cuda: ", use_cuda)
print("- Path: ", os.getcwd())
print("- PyTorch", torch.__version__)
print("- Python: ", sys.version)

n_epochs = 10
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False

# train
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', 
                             train=True, 
                             download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

# test
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', 
                             train=False, 
                             download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


# Building the Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        layer_values={}
        layer_values["input_image"]=x
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        layer_values["conv1_output"]=x
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        layer_values["conv2_output"]=x
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        layer_values["fc1_output"]=x
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        layer_values["fc2_output"]=x
        x = F.log_softmax(x)
        layer_values["output_softmax"]=x
        
        return layer_values


# Training the model
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(1, n_epochs+1)]

# model name
date_string = datetime.now().strftime("%Y-%m%d-%H%M")
model_name = "model__" + date_string
optim_name = "optim__" + date_string

def model_train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)["output_softmax"]
        train_loss = F.nll_loss(output, target)
        train_loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0 and batch_idx != 0:
            print('epoch: {} [{}/{} ]\t train loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), train_loss.item()))
            
            train_losses.append(train_loss.item())
            train_counter.append((batch_idx * batch_size_train) + ((epoch - 1)*len(train_loader.dataset)))
            
            model_path = os.path.join(os.getcwd(), "results", model_name + ".pth")
            optim_path = os.path.join(os.getcwd(), "results", optim_name + ".pth")
            torch.save(network.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optim_path)

def model_test():
    network.eval()
    test_loss = 0
    nb_correct = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data)["output_softmax"]
            test_loss += F.nll_loss(output, target, size_average=False).item()
            test_batch_loss = F.nll_loss(output, target, size_average=False).item() / batch_size_test
            # test_loss /= len(test_loader.dataset)
            # test_losses.append(test_loss)
            pred = output.data.max(1, keepdim=True)[1]
            nb_correct += pred.eq(target.data.view_as(pred)).sum()
            print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{}'.format(test_batch_loss, nb_correct, batch_size_test * (batch_idx + 1)))
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

for epoch in range(1, n_epochs + 1):
    model_train(epoch)
    model_test()

# Evaluating the Model's Performance

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.plot(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()

# Get nodes values
# https://www.aiworkbox.com/lessons/examine-mnist-dataset-from-pytorch-torchvision
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)

# train
train_loader_for_MI = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=False)

# test
test_loader_for_MI = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=False)

def train_data_for_MI():
    network.eval()
    
    list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader_for_MI):
            result = network(data)
            list.append(result)
            
    return list

def test_data_for_MI():
    network.eval()
    
    list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader_for_MI):            
            result = network(data)
            list.append(result)
            
    return list


list = train_data_for_MI()
print("input_image    : ", list[0]["input_image"].shape)
print("conv1_output   : ", list[0]["conv1_output"].shape)
print("conv2_output   : ", list[0]["conv2_output"].shape)
print("fc1_output     : ", list[0]["fc1_output"].shape)
print("fc2_output     : ", list[0]["fc2_output"].shape)
print("output_softmax : ", list[0]["output_softmax"].shape)

tensordata_input_image = []
tensordata_conv1_output = []
tensordata_conv2_output = []
tensordata_fc1_output = []
tensordata_fc2_output = []
tensordata_output_softmax = []

for i in range(len(train_loader_for_MI)):
    tensordata_input_image.append(list[i]["input_image"].data.numpy().flatten())
    tensordata_conv1_output.append(list[i]["conv1_output"].data.numpy().flatten())
    tensordata_conv2_output.append(list[i]["conv2_output"].data.numpy().flatten())
    tensordata_fc1_output.append(list[i]["fc1_output"].data.numpy().flatten())
    tensordata_fc1_output.append(list[i]["fc2_output"].data.numpy().flatten())
    tensordata_output_softmax.append(list[i]["output_softmax"].data.numpy().flatten())

tensordata_input_image = np.array(tensordata_input_image)
tensordata_conv1_output = np.array(tensordata_conv1_output)
tensordata_conv2_output = np.array(tensordata_conv2_output)
tensordata_fc1_output = np.array(tensordata_fc1_output)
tensordata_fc2_output = np.array(tensordata_fc2_output)
tensordata_output_softmax = np.array(tensordata_output_softmax)


# MINE (試作ver.)
x = tensordata_conv1_output
y = tensordata_conv2_output
z = np.concatenate([x, y], axis=1)
print("x.shape: ", x.shape)
print("y.shape: ", y.shape)
print("z.shape: ", z.shape)

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

class StochasticNet(nn.Module):
    def __init__(self, xdim=1, ydim=1, hidden_size=10000):
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim
        
        self.fc1 = nn.Linear(self.xdim+self.ydim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.normal_(self.fc4.weight, std=0.02)
        nn.init.constant_(self.fc4.bias, 0)
        nn.init.normal_(self.fc5.weight, std=0.02)
        nn.init.constant_(self.fc5.bias, 0)
        
    def forward(self, input_):
        output = F.relu(self.fc1(input_))
        output = F.dropout(output, p=0.2, training=True)
        output = F.relu(self.fc2(output))
        output = F.dropout(output, p=0.2, training=True)
        output = F.relu(self.fc3(output))
        output = F.dropout(output, p=0.2, training=True)
        output = F.relu(self.fc4(output))
        output = F.dropout(output, p=0.2, training=True)
        output = self.fc5(output)
        return output

def calc_MI_LowerBound(joint, marginal, net):
    t = net(joint)
    et = torch.exp(net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et)) # Lower bound for MI
    return mi_lb, t, et

def update(joint_batch, marginal_batch, net, optimizer, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint_batch = torch.autograd.Variable(torch.FloatTensor(joint_batch)).cuda()
    marginal_batch = torch.autograd.Variable(torch.FloatTensor(marginal_batch)).cuda()
    
    mi_lb , t, et = calc_MI_LowerBound(joint_batch, marginal_batch, net)
    
    # unbiasing use moving average
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    loss = -1 * (torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et)) # original loss function
    # use biased estimator
    # loss = - mi_lb
    
    optimizer.zero_grad()
    autograd.backward(loss)
    optimizer.step()
    
    return mi_lb, ma_et, loss

def train(x, y, net, optimizer, batch_size=1000, nb_epoch=int(5e+4), log_freq=int(5e+2)):

    mi_lower_bounds = []
    losses = []
    # moving average of exp(T)
    ma_et = 1.
    
    for i in range(nb_epoch):
        if x.shape[0] != y.shape[0]: 
            print("shape error.")
            break
            
        sample_size = x.shape[0]
        joint_batch = sample_batch(x, y, sample_size, batch_size=batch_size, sample_mode='joint')
        marginal_batch = sample_batch(x, y, sample_size, batch_size=batch_size, sample_mode='marginal')
        
        mi_lb, ma_et, loss = update(joint_batch, marginal_batch, net, optimizer, ma_et)
        mi_lower_bounds.append(mi_lb.detach().cpu().numpy())
        losses.append(loss)
        
        if (i + 1) % (log_freq) == 0:
            print('epoch: {:>6} \t MI lower bounds: {:2.4f} \t Loss of MINE: {:2.4f}'.format(i+1, mi_lower_bounds[-1], losses[-1]))
            
    return mi_lower_bounds, losses

# moving average
def ma(array, window_size=100):
    return [np.mean(array[i : i + window_size]) for i in range(0, len(array) - window_size)]


net = StochasticNet(xdim=x.shape[1], ydim=y.shape[1]).cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
mi_lower_bounds, losses = train(x, y, net, optimizer)

# MI lower boundsのプロット
mi_lower_bound_ma = ma(mi_lower_bounds, window_size=200)
# plt.title("Moving Average of MI(X, Y) \n  (window_size=200)")
# plt.xlabel("epoch")
# plt.ylabel("MA of MI(X, Y)")
# plt.plot(range(len(mi_lower_bound_ma)), mi_lower_bound_ma)
# plt.show()
print("Final value of MI: ", mi_lower_bound_ma[-1])

