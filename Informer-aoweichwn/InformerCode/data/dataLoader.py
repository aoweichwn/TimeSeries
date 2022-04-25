# -*- coding:utf-8 -*-
# @Time : 2022/4/17 11:57
# @Author : aoweichen
# @File : dataLoader.py
# @Software: PyCharm

import os
import numpy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from InformerCode.utils.tools import StandarScaler
from InformerCode.utils.timefeatures import time_features
"""
    过滤警告
"""
import warnings
warnings.filterwarnings("ignore")

"""

"""
class Dataset_ETT_hour(Dataset):
    def __init__(self,rootPath,flag='train',size=None,features='MS',dataPath='ETTh1.csv',target='OT',
                 scale=True,inverse=False,timeenc = 0,freq='h',cols=None):
        """
        初始化一些必要项

        :param rootPath: str类型， 数据文件的根目录
        :param flag: str类型， 表示是训练模式还是测试模式还是验证模式
        :param size: 列表，[seq_len, label_len, preb_len]。seq_len:数值，表示输入序列的长度（训练）;label_len:数值，表示验证序列的长度
                        （用于验证预测数据准确性的序列的长度）;preb_len:数值，预测序列的长度；
                        如果没有设置size的话，使用默认配置的size,size = [24*4*4,24*4,24*4]。
        :param features: str类型，timefeatures中的一个参数
        :param dataPath: str类型，表示数据文件的名称（路径，相对于数据文件根目录）
        :param target: 最终预测的目标项
        :param scale: bool类型，表示是否对数据进行标准化处理
        :param inverse:
        :param timeenc: 0或1，表示事件编码的格式
        :param freq: str类型,表示时间编码的粒度（细度，精确度）
        :param cols:
        """
        self.pred_len = None
        """
            如果没有设置size的话，使用默认配置的size
            size = [24*4*4,24*4,24*4]
        """
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        """
            设置数据集的模式：'train','test' or 'val'
        """
        assert flag in ['train','test','val'],f"The input param flag = {flag} is not in ['train','test','val']"
        typeMap = {'train':0,'test':1,'val':2}
        self.setType = typeMap[flag]

        self.features,self.target,self.scale,self.inverse,self.timeenc,self.freq,self.rootPath,self.dataPath = \
                features,target,scale,inverse,timeenc,freq,rootPath,dataPath
        # 读取数据，初始化参数
        self.__read_data__()

    def __read_data__(self):
        """

        :return:
        """
        global df_data
        # 标准化类
        self.scaler = StandarScaler()
        # 读取数据到pd.DataFrame中
        # df_raw -> 原始数据
        df_raw = pd.read_csv(os.path.join(self.rootPath,self.dataPath))

        """
        border1s:表示数据左边界（[trainLeftBorder,testLeftBorder,valLeftBorder]）
        border1s:表示数据右边界（[trainRightBorder,testRightBorder,valRightBorder]）
        border1 = border1s[self.setType]
        border2 = border2s[self.setType]
        """
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24-self.seq_len]
        border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
        border1 = border1s[self.setType]
        border2 = border2s[self.setType]

        """
        features: 表示预测任务是哪一种，作者给了三种方案，分别是：
            "M": 表示用多列数据去预测多列数据
            "S": 表示用单变量数据去预测单变量数据（一列预测一列）
            "MS": 表示用多列数据去预测单列数据
        下面的代码就是确定预测任务的种类并对数据作相应处理
        df_data => 表示需要预测的数据
        """
        if self.features == 'M' or self.features == 'MS':
            # data.columns返回一个index类型的列索引列表，data.columns.values返回的是列索引组成的ndarray类型。
            # 得到除了时间戳之外的Index
            # df_raw.columns[1:] => Index(['date','HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
            # df_raw.columns[1:] => Index(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
            # data.columns.values => ['date' 'HUFL' 'HULL' 'MUFL' 'MULL' 'LUFL' 'LULL' 'OT']
            # cols_data => Index(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
            cols_data = df_raw.columns[1:]
            # df_data => 除了时间戳之外的所有数据的DF
            # df_data 表示根据需要预测的数据所做的标签
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # [[]] 表示一个dataframe类型
            # 得到需要预测目标的数据，此处为'OT'列
            df_data = df_raw[[self.target]]

        """
            只有训练集才做数据标准化
        """
        if self.scale:
            # 得到训练集数据
            train_data = df_data[border1s[0]:border2s[0]]
            # .values => 只有DataFrame中的值会被返回，轴标签会被移除。
            # 用df_data中的一部分数据的均值和方差来对所有数据进行标准化
            self.scaler.fit(train_data.values)
            # data => 标准化后的数据
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 得到时间戳信息
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] =  pd.to_datetime(df_stamp.date)
        # 对时间进行编码，得到时间戳的特征信息
        data_stamp = time_features(df_stamp,timeenc=self.timeenc,freq=self.freq)

        # 数据data_x
        self.data_x = data[border1:border2]
        # self.inverse
        # 对于已经标准化后的数据，data_y表示标准化之前的数据
        # 或者说对于没有标准化的数据,data_y表示标准化后的数据
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        # 时间戳
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
             得到一个betch的序列数据
         """
        # 用于训练的输入数据
        s_begin = index
        s_end = s_begin + self.seq_len
        # 应该是预测标签的验证数据或者训练数据
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # s_,r_分别处理data_x和data_y的数据
        seq_x = self.data_x[s_begin:s_end]
        """
        
        """
        if self.inverse:
            # numpy提供了numpy.append(arr, values, axis=None)函数。对于参数规定，要么一个数组和一个数值；要么两个数组，
            # 不能三个及以上数组直接append拼接。
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len],self.data_y[r_begin+self.label_len:r_end]],0)
        else:
            seq_y = self.data_y[r_begin:r_end]
            # seq_x_mark->
            #           时间戳的序列信息
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len +1

    def inverse_transform(self,data):
        return self.scaler.inverse_transform(data)

"""

"""
class Dataset_ETT_minute(Dataset):
    def __init__(self, rootPath, flag='train', size=None, features='S', dataPath='ETTh1.csv', target='OT',
                 scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        """
        初始化一些必要项

        :param rootPath: str类型， 数据文件的根目录
        :param flag: str类型， 表示是训练模式还是测试模式还是验证模式
        :param size: 列表，[seq_len, label_len, preb_len]。seq_len:数值，表示输入序列的长度（训练）;label_len:数值，表示验证序列的长度
                        （用于验证预测数据准确性的序列的长度）;preb_len:数值，预测序列的长度；
                        如果没有设置size的话，使用默认配置的size,size = [24*4*4,24*4,24*4]。
        :param features: str类型，timefeatures中的一个参数
        :param dataPath: str类型，表示数据文件的名称（路径，相对于数据文件根目录）
        :param target: 最终预测的目标项
        :param scale: bool类型，表示是否对数据进行标准化处理
        :param inverse:
        :param timeenc: 0或1，表示事件编码的格式
        :param freq: str类型,表示时间编码的粒度（细度，精确度）
        :param cols:
        """
        self.pred_len = None
        """
            如果没有设置size的话，使用默认配置的size
            size = [24*4*4,24*4,24*4]
        """
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[0]
            self.pred_len = size[0]

        """
            设置数据集的模式：'train','test' or 'val'
        """
        assert flag in ['train', 'test', 'val'], f"The input param flag = {flag} is not in ['train','test','val']"
        typeMap = {'train': 0, 'test': 1, 'val': 2}
        self.setType = typeMap[flag]

        self.features, self.target, self.scale, self.inverse, self.timeenc, self.freq, self.rootPath, self.dataPath = \
            features, target, scale, inverse, timeenc, freq, rootPath, dataPath
        # 读取数据，初始化参数
        self.__read_data__()

    def __read_data__(self):
        """

        :return:
        """
        global df_data
        # 标准化类
        self.scaler = StandarScaler()
        # 读取数据到pd.DataFrame中
        # df_raw -> 原始数据
        df_raw = pd.read_csv(os.path.join(self.rootPath,self.dataPath))

        """
        border1s:表示数据左边界（[trainLeftBorder,testLeftBorder,valLeftBorder]）
        border1s:表示数据右边界（[trainRightBorder,testRightBorder,valRightBorder]）
        border1 = border1s[self.setType]
        border2 = border2s[self.setType]
        """
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.setType]
        border2 = border2s[self.setType]

        """
        features: 表示预测任务是哪一种，作者给了三种方案，分别是：
            "M": 表示用多列数据去预测多列数据
            "S": 表示用单变量数据去预测单变量数据（一列预测一列）
            "MS": 表示用多列数据去预测单列数据
        下面的代码就是确定预测任务的种类并对数据作相应处理
        df_data => 表示需要预测的数据
        """
        if self.features == 'M' or self.features == 'MS':
            # data.columns返回一个index类型的列索引列表，data.columns.values返回的是列索引组成的ndarray类型。
            # 得到除了时间戳之外的Index
            # df_raw.columns[1:] => Index(['date','HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
            # df_raw.columns[1:] => Index(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
            # data.columns.values => ['date' 'HUFL' 'HULL' 'MUFL' 'MULL' 'LUFL' 'LULL' 'OT']
            # cols_data => Index(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
            cols_data = df_raw.columns[1:]
            # df_data => 除了时间戳之外的所有数据的DF
            # df_data 表示根据需要预测的数据所做的标签
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # [[]] 表示一个dataframe类型
            # 得到需要预测目标的数据，此处为'OT'列
            df_data = df_raw[[self.target]]

        """
            只有训练集才做数据标准化
        """
        if self.scale:
            # 得到训练集数据
            train_data = df_data[border1s[0]:border2s[0]]
            # .values => 只有DataFrame中的值会被返回，轴标签会被移除。
            # 用df_data中的一部分数据的均值和方差来对所有数据进行标准化
            self.scaler.fit(train_data.values)
            # data => 标准化后的数据
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 得到时间戳信息
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # 对时间进行编码，得到时间戳的特征信息
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # 数据data_x
        self.data_x = data[border1:border2]
        # self.inverse
        # 对于已经标准化后的数据，data_y表示标准化之前的数据
        # 或者说对于没有标准化的数据,data_y表示标准化后的数据
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        # 时间戳
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
             得到一个betch的序列数据
         """
        # 用于训练的输入数据
        s_begin = index
        s_end = s_begin + self.seq_len
        # 应该是预测标签的验证数据或者训练数据
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # s_,r_分别处理data_x和data_y的数据
        seq_x = self.data_x[s_begin:s_end]
        """

        """
        if self.inverse:
            # numpy提供了numpy.append(arr, values, axis=None)函数。对于参数规定，要么一个数组和一个数值；要么两个数组，
            # 不能三个及以上数组直接append拼接。
            seq_y = np.concatenate([self.data_x[r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]],
                                   0)
        else:
            seq_y = self.data_y[r_begin, r_end]
            # seq_x_mark->
            #           时间戳的序列信息
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.preb_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




"""
下面两个要看，没咋看
"""
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandarScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandarScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == "__main__":
    # data = pd.read_csv("F:\AI\DL\TimeSeries\Informer\InformerCode\data\ETT\ETTh1.csv")
    dataset = Dataset_ETT_hour("F:\AI\DL\TimeSeries\Informer\InformerCode\data\ETT")
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
    print(len(dataset))
    data = DataLoader(dataset,batch_size=16)
    print(8161/16)
    for u,(batch_x,_,_,_) in enumerate(data):
        print(batch_x.shape)


































































































