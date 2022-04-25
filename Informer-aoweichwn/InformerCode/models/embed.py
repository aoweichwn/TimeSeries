# -*- coding:utf-8 -*-
# @Time : 2022/4/22 18:47
# @Author : aoweichen
# @File : embed.py
# @Software: PyCharm
import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Informer中注意力机制的位置编码
    """

    def __init__(self,dMOdel,maxLen = 5000):
        """

        :param dMOdel: 输入特征的维度
        :param maxLen: 序列的最大长度（初始化为0）
        """
        super(PositionalEmbedding, self).__init__()

        # PE => [maxLen,dModel]
        PE = torch.zeros(maxLen,dMOdel).float()
        PE.require_grad = False

        # torch.arange(0,max_len) -> [0,1,...,max_len-1]
        # position.unsqueeze(1) -> [0,1,2,...,max_len-1]=>[[0],[1],...,[max_len-1]]
        position = torch.arange(0,maxLen).float().unsqueeze(1)
        # 位置编码中的公共项
        div_term = (torch.arange(0,dMOdel,2).float()*(-(math.log(10000.0) / dMOdel))).exp()

        PE[:,0::2] = torch.sin(position*div_term)
        PE[:,1::2] = torch.cos(position*div_term)

        # PyTorch中定义模型时，有时候会遇到self.register_buffer('name', Tensor)的操作，该方法的作用是定义一组参数，
        # 该组参数的特别之处在于：模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        PE = PE.unsqueeze(0)
        self.register_buffer("PE",PE)


    def forward(self,input):
        # reture => [maxLen,dModel] -> ([1,5000,512])
        return self.PE[:,:input.size(1)]

class TokenEmbdding(nn.Module):
    """
    对除时间数据以外的数据进行特征编码或者特征提取到指定维度
    """
    def __init__(self,InputChannel,dModel):
        """
        对除时间数据以外的数据进行特征编码或者特征提取到指定维度
        :param InputChannel: 输入特征的维度（一般是序列长度）
        :param dModel: 输出特征的维度（一般dModel==512）
        """
        super(TokenEmbdding, self).__init__()
        # padding_mode='circular'=>表示循环使用原始数据进行填充
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=InputChannel,out_channels=dModel,kernel_size=3,padding=padding,
                                   padding_mode='circular')
        for model in self.modules():
            if isinstance(model,nn.Conv1d):
                """
                对一维卷积的参数进行初始化
                Xavier在tanh中表现的很好，但在Relu激活函数中表现的很差，所何凯明提出了针对于Relu的初始化方法。
                Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification He, K. et al.
                 (2015)
                该方法基于He initialization,其简单的思想是：
                在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0，所以，要保持方差不变，只需要在 Xavier 的基础上再除以2,也就是说
                在方差推到过程中，式子左侧除以2.
                """
                nn.init.kaiming_normal_(model.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self,input):
        # 这里不加这句话一直报错
        input = input.float()
        # input => [Batch,Length,DataItem] => [Batch,DataItem,Length] => [Batch,dModel,DataItem]
        # 因为nn.Conv1d()是在最后一个维度上做的变换，所以要先变换维度
        # 应该是对中间那层做处理
        output = self.tokenConv(input.permute(0,2,1)).transpose(1,2)
        # output => [Batch,dModel,DataItem]
        return output

class FixedEmbedding(nn.Module):
    def __init__(self,inputDim,dModel):
        """

        :param inputDim:
        :param dModel:

        """
        super(FixedEmbedding, self).__init__()
        # 位置编码
        w = torch.zeros(inputDim, dModel).float()
        w.require_grad = False
        position = torch.arange(0, inputDim).float().unsqueeze(1)
        div_term = (torch.arange(0, dModel, 2).float() * -(math.log(10000.0) / dModel)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        # 创建inputDim个维度大小为dModel的嵌入向量
        self.emb = nn.Embedding(inputDim, dModel)
        # nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter，并且会向宿主模型注册该参数，成为一部分。
        # 即model.parameters()会包含这个parameter。从而，在参数优化的时候可以自动一起优化，这就不需要我们单独对这个参数进行优化啦。
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self,input):
        # detach()=>当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，
        # 并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播
        return self.emb(input).detach()

class TemporalEmbedding(nn.Module):
    """"""

    def __init__(self,dModel,embedType='fixed',freq='h'):
        """

        :param dModel:
        :param embedType:
        :param freq:
        """
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embedType=='fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, dModel)
        self.hour_embed = Embed(hour_size, dModel)
        self.weekday_embed = Embed(weekday_size, dModel)
        self.day_embed = Embed(day_size, dModel)
        self.month_embed = Embed(month_size, dModel)

    def forward(self, input):
        input = input.long()
        minute_input = self.minute_embed(input[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_input = self.hour_embed(input[:, :, 3])
        weekday_input = self.weekday_embed(input[:, :, 2])
        day_input = self.day_embed(input[:, :, 1])
        month_input = self.month_embed(input[:, :, 0])
        return hour_input + weekday_input + day_input + month_input + minute_input

class TimeFeatureEmbedding(nn.Module):
    """

    """
    def __init__(self,dModel,embdeType='timeF',freq='h'):
        """

        :param dModel:
        :param embdeType:
        :param freq:
        """
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        dimInput = freq_map[freq]
        # PyTorch的nn.Linear（）是用于设置网络中的全连接层的，需要注意在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，
        # 形状通常为[batch_size, size]，不同于卷积层要求输入输出是四维张量。
        self.embed = nn.Linear(dimInput,dModel)

    def forward(self,input):
        return self.embed(input)

class DataEmbedding(nn.Module):
    """

    """
    def __init__(self,inputDim,dModel,embedType = 'fixed',freq='h',dropout = 0.1):
        """

        :param inputDim:
        :param dModel:
        :param embedType:
        :param freq:
        :param dropout:
        """
        super(DataEmbedding, self).__init__()
        # 值编码
        # [batch,Length,]
        self.value_embedding = TokenEmbdding(inputDim,dModel)
        # 位置编码
        self.position_embedding = PositionalEmbedding(dModel)
        # 时间戳编码
        self.temporal_embedding = TemporalEmbedding(dModel,embedType=embedType,freq=freq) if embedType != 'timeF' \
            else TimeFeatureEmbedding(dModel,embedType,freq)
        # 随机dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,input,inputMark):
        output = self.value_embedding(input) + self.position_embedding(input) + self.temporal_embedding(inputMark)
        return self.dropout(output)


if __name__ == "__main__":
    pass