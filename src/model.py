import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import config

def calculate_output_size(input_size, kernel_size, stride, num_convs):
    for i in range(num_convs):
        # Apply convolution
        input_size = (input_size - (kernel_size[i] - 1) - 1) // stride[i] + 1
        # Apply max pooling
        # input_size = (input_size - (pool_kernel_size - 1) - 1) // pool_stride + 1
    return input_size


class cnn_block(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super(cnn_block, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv1d(inp, oup, kernel_size, stride,padding=1),
                                    nn.BatchNorm1d(oup),
                                    nn.ReLU())
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            x = self.conv(x)
        return x


class net_cnn(nn.Module):
    def __init__(self, input_size=8192, num_classes=4):
        super(net_cnn, self).__init__()
        self.conv_layer = []
        for i in range(len(config.out_channel_list)-1):
            self.conv_layer.append(cnn_block(config.out_channel_list[i], config.out_channel_list[i+1], config.kernel_size[i], config.stride[i]))
          
        self.net_conv1d = nn.Sequential(*self.conv_layer)
        # final_output_size = calculate_output_size(input_size, config.kernel_size, config.stride, len(config.out_channel_list)-1)

        self.fc1 = nn.Linear(40960, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net_conv1d(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Apply softmax to output probabilities
        x = nn.LogSoftmax(dim=1)(x)
        return x
