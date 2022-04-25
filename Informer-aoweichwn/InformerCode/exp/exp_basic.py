# -*- coding:utf-8 -*-
# @Time : 2022/4/25 8:41
# @Author : aoweichen
# @File : exp_basic.py
# @Software: PyCharm

import os
import torch
import numpy as np

# 例子集的基类
class EXP_Basic(object):
    """

    """
    def __init__(self,args):
        """

        :param args:
        """
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self,flag):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self):
        pass

    def test(self):
        pass