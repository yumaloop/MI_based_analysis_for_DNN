import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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


