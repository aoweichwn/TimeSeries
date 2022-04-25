# -*- coding:utf-8 -*-
# @Time : 2022/4/23 17:33
# @Author : aoweichen
# @File : decoder.py
# @Software: PyCharm

import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """

    """
    def __init__(self, selfAttention,crossAttention,dModel,d_ff,dropout = 0.1,activation="relu"):
        """

        :param selfAttention:
        :param crossAttention:
        :param dModel:
        :param d_ff:
        :param dropout:
        :param activation:
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*dModel
        self.selfAttention = selfAttention
        self.crossAttention = crossAttention
        self.conv1 = nn.Conv1d(in_channels=dModel, out_channels=d_ff,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=dModel, kernel_size=1)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.norm3 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, input, cross, inputMask=None,crossMask=None):
        """

        :param input: [Batch,Length,DModel]
        :param cross:
        :param inputMask:
        :param crossMask:
        :return:
        """
        # input = [Batch,Length,DModel]
        input = input + self.dropout(self.selfAttention(
            input,input,input,
            attnMask = inputMask
        )[0])
        # input = [Batch,Length,DModel]
        input = self.norm1(input)

        # 这里的是encoder和decoder交叉那部分的注意力
        # input = [Batch,Length,DModel]
        input = input + self.dropout(self.crossAttention(
            input,cross,cross,
            attnMask = crossMask
        )[0])
        # output = [Batch,Length,DModel]
        output = input = self.norm2(input)
        output = self.dropout(self.activation(self.conv1(output.transpose(-1,1))))
        output = self.dropout(self.conv2(output).transpose(-1,1))

        output = self.norm3(input + output)
        # output = [Batch,Length,DModel]
        return output

class Decoder(nn.Module):
    """

    """

    def __init__(self,layers,normLayer=None):
        """

        :param layers:
        :param normLayer:
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = normLayer

    def forward(self, input, cross, inputMask,crossMask):
        """

        :param input: [Batch,Length,DModel]
        :param cross:
        :param inputMask:
        :param crossMask:
        :return:
        """

        for layer in self.layers:
            input = layer(input,cross,inputMask,crossMask)

        if self.norm is not None:
            output = self.norm(input)
        else:
            output = input
        # output = [Batch,Length,DModel]
        return output