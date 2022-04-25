# -*- coding:utf-8 -*-
# @Time : 2022/4/23 17:33
# @Author : aoweichen
# @File : model.py
# @Software: PyCharm

import torch
import torch.nn as nn
from InformerCode.models.encoder import Encoder,EncoderLayer,ConvLayer,EncoderStacks
from InformerCode.models.decoder import Decoder,DecoderLayer
from InformerCode.models.attn import FullAttention,ProbAttention,AttentionLayer
from InformerCode.models.embed import DataEmbedding

class Informer(nn.Module):
    """

    """
    def __init__(self,encInputDim,decInputDim,outputDim,seqLen,labelLen,outLen,factor=5,dModel=512,nHeads=8,
                 encLayers=3,decLayers=2,d_ff=512,dropout=0.0,attn='prob',embed='fixed',freq='h',activation='gelu',
                 outputAttention = False,distil=True,mix=True,device=torch.device('cuda:0')):
        """

        :param encInputDim:
        :param decInputDim:
        :param outputDim:
        :param seqLen:
        :param labelLen:
        :param outLen:
        :param factor:
        :param dModel:
        :param nHeads:
        :param encLayers:
        :param decLayers:
        :param d_ff:
        :param dropout:
        :param attn:
        :param embed:
        :param freq:
        :param activation:
        :param outputAttention:
        :param distil:
        :param mix:
        :param device:
        """
        super(Informer, self).__init__()
        self.predLen = outLen
        self.attn = attn
        self.outputAttention = outputAttention

        # 数据编码部分
        self.encEmbedding = DataEmbedding(encInputDim,dModel,embed,freq,dropout)
        self.decEmbedding = DataEmbedding(decInputDim,dModel,embed,freq,dropout)

        # 注意力机制部分
        Attn = ProbAttention if attn == "prob" else FullAttention

        # 编码器部分
        self.encoder = Encoder(
            # attn部分
            [
                EncoderLayer(
                    AttentionLayer(Attn(False,factor,attentionDropout=dropout,outputAttention=outputAttention),
                                   dModel,nHeads,mix=False),
                    dModel,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(encLayers)
            ],
            # conv部分,这里说明conv比attn少一层(distil)
            [
                ConvLayer(
                    dModel
                ) for _ in range(encLayers - 1)
            ] if distil else None,
            normLayer=nn.LayerNorm(dModel)
        )

        # 解码器部分
        # crosss部分采用原始attn方法
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True,factor,attentionDropout=dropout,outputAttention=False),
                                   dModel,nHeads,mix=mix),
                    AttentionLayer(FullAttention(False,factor,attentionDropout=dropout,outputAttention=False),
                                   dModel,nHeads,mix=False),
                    dModel,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(decLayers)
            ],
            #
            normLayer=nn.LayerNorm(dModel)
        )

        #
        self.projection = nn.Linear(dModel,outputDim,bias=True)

    def forward(self,inputEncEmbeddingData,inputMarkEncEmbeddingData,inputDecEmbeddingData,inputMarkDecEmbeddingData,
                encSelfMask=None,decSelfMask=None,decEncMask=None):
        """

        :param inputEncEmbeddingData:
        :param inputMarkEncEmbeddingData:
        :param inputDecEmbeddingData:
        :param inputMarkDecEmbeddingData:
        :param encSelfMask:
        :param decSelfMask:
        :param decEncMask:
        :return:
        """
        # 编码器部分
        encEmbedding = self.encEmbedding(inputEncEmbeddingData,inputMarkEncEmbeddingData)
        encOutput,attns = self.encoder(encEmbedding,attnMask=encSelfMask)

        # 解码器部分
        # 从这里的代码基本上可以看出来，解码器只有一层
        decEmbedding = self.decEmbedding(inputDecEmbeddingData,inputMarkDecEmbeddingData)
        decOutput = self.decoder(decEmbedding,encEmbedding,inputMask=decSelfMask,crossMask=decEncMask)
        decOutput = self.projection(decOutput)


        if self.outputAttention:
            return decOutput[:,-self.predLen:,:],attns
        else:
            # [Batch,Len,Dim]
            # [32,72,7]
            return decOutput[:,-self.predLen:,:]

class InformerStack(nn.Module):
    """

    """

    def __init__(self,encInputDim,decInputDim,outputDim,seqLen,labelLen,outLen,factor=5,dModel=512,nHeads=8,
                 encLayers=[3,2,1],decLayers=2,d_ff=512,dropout=0.0,attn='prob',embed='fixed',freq='h',activation='gelu',
                 outputAttention = False,distil=True,mix=True,device=torch.device('cuda:0')):
        """

        :param encInputDim:
        :param decInputDim:
        :param outputDim:
        :param seqLen:
        :param labelLen:
        :param outLen:
        :param factor:
        :param dModel:
        :param nHeads:
        :param encLayers:
        :param decLayers:
        :param d_ff:
        :param dropout:
        :param attn:
        :param embed:
        :param freq:
        :param activation:
        :param outputAttention:
        :param distil:
        :param mix:
        :param device:
        """
        super(InformerStack, self).__init__()
        self.predLen = outLen
        self.attn = attn
        self.outputAttention = outputAttention

        # 数据编码部分
        self.encEmbedding = DataEmbedding(encInputDim,dModel,embed,freq,dropout)
        self.decEmbedding = DataEmbedding(decInputDim,dModel,embed,freq,dropout)

        # 注意力机制部分
        Attn = ProbAttention if attn == "prob" else FullAttention

        # 编码器部分
        inputLens = list(range(len(encLayers)))
        encoders = [
            Encoder(
                # attn部分
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attentionDropout=dropout, outputAttention=outputAttention),
                                       dModel, nHeads, mix=False),
                        dModel,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(encLayer)
                ],
                # conv部分,这里说明conv比attn少一层(distil)
                [
                    ConvLayer(
                        dModel
                    ) for _ in range(encLayer - 1)
                ] if distil else None,
                normLayer=nn.LayerNorm(dModel)
            ) for encLayer in encLayers
        ]
        self.encoder = EncoderStacks(encoders,inputLength=inputLens)

        # 解码器部分
        # crosss部分采用原始attn方法
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True,factor,attentionDropout=dropout,outputAttention=False),
                                   dModel,nHeads,mix=mix),
                    AttentionLayer(FullAttention(False,factor,attentionDropout=dropout,outputAttention=False),
                                   dModel,nHeads,mix=False),
                    dModel,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(decLayers)
            ],
            #
            normLayer=nn.LayerNorm(dModel)
        )
        self.projection = nn.Linear(dModel,outputDim,bias=True)

    def forward(self,inputEncEmbeddingData,inputMarkEncEmbeddingData,inputDecEmbeddingData,inputMarkDecEmbeddingData,
                encSelfMask=None,decSelfMask=None,decEncMask=None):
        """

        :param inputEncEmbeddingData:
        :param inputMarkEncEmbeddingData:
        :param inputDecEmbeddingData:
        :param inputMarkDecEmbeddingData:
        :param encSelfMask:
        :param decSelfMask:
        :param decEncMask:
        :return:
        """
        # 编码器部分
        encEmbedding = self.encEmbedding(inputEncEmbeddingData,inputMarkEncEmbeddingData)
        encOutput,attns = self.encoder(encEmbedding,attnMask=encSelfMask)

        # 解码器部分
        # 从这里的代码基本上可以看出来，解码器只有一层
        decEmbedding = self.decEmbedding(inputDecEmbeddingData,inputMarkDecEmbeddingData)
        decOutput = self.decoder(decEmbedding,encEmbedding,inputMask=decSelfMask,crossMask=decEncMask)
        decOutput = self.projection(decOutput)


        if self.outputAttention:
            return decOutput[:,-self.predLen:,:],attns
        else:
            # [Batch,Len,Dim]
            # [32,72,7]
            return decOutput[:,-self.predLen:,:]

























