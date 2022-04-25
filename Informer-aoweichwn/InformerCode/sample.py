# -*- coding:utf-8 -*-
# @Time : 2022/4/22 21:45
# @Author : aoweichen
# @File : sample.py
# @Software: PyCharm
from data.dataLoader import Dataset_ETT_hour
from models.embed import DataEmbedding
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    dataest = Dataset_ETT_hour("F:\AI\DL\TimeSeries\Informer\InformerCode\data\ETT",size=[96,48,24])
    dataLoader = DataLoader(dataest,batch_size=32,shuffle=True,drop_last=True)
    for u,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(dataLoader):
            enc_embedding = DataEmbedding(7,512)
            enc_out = enc_embedding(batch_y,batch_y_mark)
            print(enc_out.shape)
