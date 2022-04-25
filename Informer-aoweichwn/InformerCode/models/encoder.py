# -*- coding:utf-8 -*-
# @Time : 2022/4/23 17:33
# @Author : aoweichen
# @File : encoder.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from InformerCode.models.attn import AttentionLayer,ProbAttention

class ConvLayer(nn.Module):
    """

    """
    def __init__(self,inChannels):
        super(ConvLayer, self).__init__()
        padding = 1
        self.downConv = nn.Conv1d(in_channels=inChannels,out_channels=inChannels,
                                  kernel_size=3,padding=padding,padding_mode='circular')
        self.norm = nn.BatchNorm1d(inChannels)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)

    def forward(self, input):
        """

        :param input: [Batch,Length,DModel]
        :return:
        """
        # input => [Batch, DModel, Length]
        input = self.downConv(input.permute(0,2,1))
        # input => [Batch, DModel, Length]
        input = self.norm(input)
        # input = [Batch,DModel,Length]
        input = self.activation(input)
        # input = [Batch,DModel,Length/2]
        input = self.maxPool(input)
        # output = [Batch,Length/2,DModel]
        output = input.transpose(1,2)
        return output

class EncoderLayer(nn.Module):
    """

    """
    def __init__(self,attention,dModel,d_ff=None,dropout=0.1,activation='relu'):
        """

        :param attention:
        :param dModel:
        :param d_ff: Dimension of fcn (defaults to 2048)
        :param dropout:
        :param activation:
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*dModel
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=dModel,out_channels=d_ff,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=dModel,kernel_size=1)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self,input,attnMask = None):
        """

        :param input: [Batch,Length,DModel]
        :param attnMask: bool
        :return: output => [Batch,Length,dModel],attn
        """

        # newInput = [Batch,Length,DModel]
        newInput ,attn = self.attention(
           input,input,input,
            attnMask=attnMask
        )
        # input = [Batch,Length,DModel]
        input = input + self.dropout(newInput)

        # output = input => [Batch,Length,DModel]
        output = input = self.norm1(input)
        # output => [Batch,d_ff,Length]
        output = self.dropout(self.activation(self.conv1(output.transpose(-1,1))))
        # output => [Batch,Length,dModel]
        output = self.dropout(self.conv2(output).transpose(-1,1))
        # output => [Batch,Length,dModel]
        output = self.dropout(self.norm2(input + output))
        return output, attn

class Encoder(nn.Module):
    """

    """
    def __init__(self,attnLayers,convLayers=None,normLayer=None):
        """

        :param attnLayers:
        :param convLayers:
        :param normLayer:
        """
        super(Encoder, self).__init__()
        self.attnLayers = nn.ModuleList(attnLayers)
        self.convLayers = nn.ModuleList(convLayers)
        self.norm = normLayer

    def forward(self, input, attnMask):
        """

        :param input: [Batch,Length,DModel]
        :param attnMask:
        :return: output = [batch,Length/(2*n),DModel], attns
        """
        attns = []
        if self.convLayers is not None:
            # 一般来说，只有一层
            for attnLayer, convLayer in zip(self.attnLayers,self.convLayers):
                input ,attn = attnLayer(input,attnMask=attnMask)
                input = convLayer(input)
                attns.append(attn)
            # 这一段是encoder最后一层，并没有做convLayer处理
            output,attn = self.attnLayers[-1](input,attnMask)
            attns.append(attn)
        else:
            for attnLayer in self.attnLayers:
                input, attn = attnLayer(input, attnMask=attnMask)
                attns.append(attn)

        if self.norm is not None:
            output  = self.norm(input)
        else:
            output = input
        # output = [batch,Length/(2*n),DModel]
        return output, attns

class EncoderStacks(nn.Module):
    """

    """
    def __init__(self,encoders,inputLength):
        """

        :param encoders:
        :param inputLength:
        """
        super(EncoderStacks, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inputLength = inputLength

    def forward(self, input, attnMask=None):
        """

        :param input: [Batch,96,DModel]
        :param attnMask:
        :return:outputStack = [Batch,72,DModel],attns
        """
        inputStack, attns = [], []
        for iLength, encoders in zip(self.inputLength,self.encoders):
            inputLen = input.shape[1] // (2**iLength)
            iStack, attn = encoders(input[:,-inputLen:,:])
            inputStack.append(iStack),attns.append(attn)
        outputStack = torch.cat(inputStack,-2)
        # outputStack = [Batch,72,DModel]
        return outputStack, attns





# if __name__ == "__main__":
#     conv = ConvLayer(512)
#     pro = ProbAttention()
#     attn = AttentionLayer(pro,512,6,48,48)
#     enc = EncoderLayer(attn,512)
#     a = torch.rand([32,96,512])
#     con = conv(a)
#     print(con.shape)
#     out,_ = enc(con)
#     print(out.shape)

