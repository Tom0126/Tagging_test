#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/3 00:30
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : mlp.py
# @Software: PyCharm


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Data.loader import data_loader_gnn







class MLP(nn.Module):

    def __init__(self, global_vector_channel: int, num_classes: int, dropout: bool):


        super().__init__()
        self.global_vector_channel = global_vector_channel



        # global vector
        self.linear3 = nn.Linear(global_vector_channel, 128)
        self.bn8 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(p=dropout)

        self.linear4 = nn.Linear(128, 256)
        self.bn9 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(p=dropout)

        # after cat
        self.linear5 = nn.Linear(256, 128)
        self.bn10 = nn.BatchNorm1d(128)
        self.dp5 = nn.Dropout(p=dropout)

        self.linear6 = nn.Linear(128, num_classes)

    def forward(self, points, global_vector):

        batch_size= global_vector.shape[0]

        global_vector = self.linear3(global_vector)

        if batch_size > 1:
            global_vector = self.bn8(global_vector)

        global_vector = F.leaky_relu(global_vector, negative_slope=0.2)   # (batch_size, global_vector) -> (batch_size, 128)
        global_vector = self.dp3(global_vector)

        global_vector = self.linear4(global_vector)
        if batch_size > 1:
            global_vector = self.bn9(global_vector)
        global_vector = F.leaky_relu(global_vector,
                                     negative_slope=0.2)  # (batch_size, 128) -> (batch_size, 256)
        x = self.dp4(global_vector)





        x = self.linear5(x)
        if batch_size > 1:
            x = self.bn10(x)
        x = F.leaky_relu(x, negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp5(x)


        x = self.linear6(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x


if __name__ == '__main__':

    dgcnn = MLP(
                 global_vector_channel=2,

                 dropout=0.5,
                 num_classes=2
                             )

    test_t = torch.Tensor(np.zeros((8, 2)))

    tr=dgcnn(test_t)
    print(tr)
    summary(dgcnn,  (2,)) #23.38 MB
    pass
