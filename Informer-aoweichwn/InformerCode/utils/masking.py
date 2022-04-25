# -*- coding:utf-8 -*-
# @Time : 2022/4/23 16:04
# @Author : aoweichen
# @File : masking.py
# @Software: PyCharm
import torch

class TriangularCausalMask():
    """
    mask机制
    """
    def __init__(self,B,L,device="cpu"):
        mask_shape = [B,1,L,L]
        with torch.no_grad():
            # torch.triu()这个函数的目的在于返回矩阵上三角部分，其余部分定义为0。
            # diagonal(int,optional)-表明要考虑哪个对角线。
            self._mask = torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=1).to(device)

    @property
    # property 只能讀取的屬性特性
    def mask(self):
        return self._mask

class ProbMask():
    """

    """
    def __init__(self,B,H,L,index,scores,device='cpu'):
        """

        :param B:
        :param H:
        :param L:
        :param index: M_top
        :param scores: Q_K => [B,H,n_top,L_K]
        :param device:
        """
        # _mask => [L,L_K]
        _mask = torch.ones(L,scores.shape[-1],dtype=torch.bool).to(device).triu(1)
        # _mask_ex => [B,H,L,L_K]
        _mask_ex = _mask[None,None,:].expand(B,H,L,scores.shape[-1])
        # indicator => [B,H,n_top,L_K]
        indicator = _mask_ex[torch.arange(B)[:,None,None],
                             torch.arange(H)[None,:,None],
                             index,
                             :
                            ]
        # self._mask => [B,H,n_top,L_K]
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """

        :return: self._mask => [B,H,n_top,L_K]
        """
        return self._mask