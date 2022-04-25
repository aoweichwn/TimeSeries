# -*- coding:utf-8 -*-
# @Time : 2022/4/19 22:32
# @Author : aoweichen
# @File : attn.py
# @Software: PyCharm
from math import sqrt
import torch
import torch.nn as nn
import numpy as np
from InformerCode.utils.masking import ProbMask, TriangularCausalMask

"""
已检查，无问题
"""

"""
没啥问题
"""
class FullAttention(nn.Module):
    def __init__(self ,maskFlag=True, factor=5, scale=None,attentionDropout=0.1,outputAttention=False):
        """
        最初的那个attention机制的实现

        :param maskFlag:bool,表示是否做mask处理
        :param factor:
        :param scale:
        :param attentionDropout:
        :param outputAttention:
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.maskFlag = maskFlag
        self.outputAttention = outputAttention
        self.dropout = nn.Dropout(attentionDropout)

    def forward(self,queries, keys, values,attnMask):
        """

        :param queries: 查询向量Q = [B, L, H, E],其中E表示每个query数据的维度（一般是512维）
        :param keys: 键K = [B, S, H, D]
        :param values: 值V = [B, S, H, D]
        :param attnMask:
        :return:
        """
        # BatchSize, Length, Heads(multiHeads的数量), EmbddingSize(64 == 512/8)做了一下数据处理
        B, L, H, E = queries.shape
        # BatchSize, , Heads(multiHeads的数量), EmbddingSize(64 == 512/8)做了一下数据处理
        _, S, _, D = values.shape
        # 公式里面的那个scale
        scale = self.scale or 1./sqrt(E)
        # Q*KT, 这玩意不用看先，一个记号，先记下来会用就行
        scores = torch.einsum('blhe, bshd -> bhls',queries,keys)

        if self.maskFlag:
            if attnMask is None:
                attnMask = TriangularCausalMask(B,L,device=queries.device)
            scores.masked_fill_(attnMask.mask,-np.inf)

        # 看公式即可
        A = self.dropout(torch.softmax(scale*scores,dim=-1))
        V = torch.einsum("bhls,bshd->blhd",A,values)

        if self.outputAttention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

"""

"""
class ProbAttention(nn.Module):
    """

    """
    def __init__(self, maskFlag=True, factor=5, scale=None, attentionDropout=0.1, outputAttention=False):
        """

        :param maskFlag:
        :param factor:
        :param scale:
        :param attentionDropout:
        :param outputAttention:
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.maskFlag = maskFlag
        self.outputAttention = outputAttention
        self.dropout = nn.Dropout(attentionDropout)

    def _prob_QK(self,Q,K,sample_k,n_top):
        """


        :param Q:
        :param K:
        :param sample_k: 采样的queries的数量u_part => cln(L_k)
        :param n_top: c*ln(L_q)
        :return:  Q_K => [B,H,n_top,L_K], M_top => [B,H,n_top(index)]
        """

        # Q[B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # k_expand = [B,H,L_Q,L_K,E]
        K_expand = K.unsqueeze(-3).expand(B,H,L_Q,L_K,E)
        # index_sample 得到随机采样的L_Q*sample_k个索引值(在L_Q*L_K)中选择
        # 真正的u = u_part(c*ln(L_k))*L_q
        # torch.randint(L_K,(L_Q,sample_k)) => [L_Q,sample_k] 值在[0，L_K)之间的整数值
        index_sample = torch.randint(L_K,(L_Q,sample_k))
        # 这里的操作可以背下来 => [B,H,L_Q,sample_k,E]
        K_sample = K_expand[:,:,torch.arange(L_Q).unsqueeze(1),index_sample,:]
        # Q.unsqueeze(-2) => [B,H,L_Q,1,E], K_sample => [B,H,L_Q,E,sample_k]
        # Q_K_sample => [B,H,L_Q,1,sample_k] => [B,H,L_Q,sample_k]
        # Q_K_sample 即为Q*K的转置
        Q_K_sample = torch.matmul(Q.unsqueeze(-2),K_sample.transpose(-2,-1)).squeeze(-2)

        # 利用sparisty找到 Top_k的queries
        # 计算max-mean measurement
        # torch.max()[0]， 只返回最大值的每个数
        # Q_K_sample.max(-1)[0] => [B,H,L_Q], torch.div(Q_K_sample.sum(-1),L_K) => [B,H,L_Q]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1),L_K)
        # M.topk(n_top,sorted=False)[1]返回这n个值的索引
        # M_top => [B,H,n_top(index)]
        M_top = M.topk(n_top,sorted=False)[1]
        # torch.arange(B)[:,None,None] => [B,1,1]
        # 这种操作是一种技巧，可以记下来
        Q_reduce = Q[torch.arange(B)[:,None,None],
                      torch.arange(H)[None,:,None],
                      M_top,
                      :
                   ]
        # Q_reduce => [B,H,n_top,E], K => [B, H, E, L_K]
        # Q_K => [B,H,n_top,L_K]
        # 用the reduced Q to calculate Q_K
        Q_K = torch.matmul(Q_reduce,K.transpose(-2,-1))
        # Q_K => [B,H,n_top,L_K], M_top => [B,H,n_top(index)]
        return Q_K, M_top

    def _get_initial_context(self,V,L_Q):
        """
        不太懂
        :param V:
        :param L_Q:
        :return:
        """

        B,H,L_V,D = V.shape
        if not self.maskFlag:
            # V_sum => [B,H,D]
            # ???
            V_sum = V.mean(dim=-2)
            # context => [B,H,L_Q,D]
            context = V_sum.unsqueeze(-2).expand(B,H,L_Q,V_sum.shape[-1]).clone()
        else:
            assert(L_Q==L_V),f"requires that L_Q == L_V, i.e. for self-attention only"
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self,context_in,V,scores,index,L_Q,attn_mask):
        """
        这段代码里面attns特么全是softmax(Q_K/d**0.5),根本没有v，作者这么写良心不会痛吗，欺负我们大老实人
        :param context_in: 严格来说这个才算是attn
        :param V:
        :param scores: Q_K
        :param index: M_top
        :param L_Q:
        :param attn_mask:
        :return:
        """
        # # Q_K => [B,H,n_top,L_K], M_top => [B,H,n_top(index)]
        B, H, L_V, D = V.shape

        if self.maskFlag:
            # 得到掩码矩阵
            # attn_mask => [b,h,n_top,L_K]
            attn_mask = ProbMask(B,H,L_Q,index,scores,device=V.device)
            # 对Q_K做掩码处理
            # masked_fill_(mask, value)
            # 掩码操作
            # 用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致。
            # scores => [B,H,n_top,L_K]
            scores.masked_fill_(attn_mask.mask,-np.inf)
        # attn => [b,h,n_top,L_K]
        attn = torch.softmax(scores,dim=-1)
        # 上下文信息
        # contetxt => [B,H,L,L_K] * [B,H,L_V,E] => [B,H,L,E]
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.outputAttention:
            # 初始化attns
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            # attns => [B,H,L,L_V],这里竟然默认L_V==L_K
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            #
            return (context_in, attns)
        else:
            return (context_in,None)

    def forward(self,queries,keys,values,attnMask):
        """

        :param queries:
        :param keys:
        :param values:
        :param attnMask:
        :return:
        """

        B,L_Q,H,D = queries.shape
        _,L_K,_,_ = keys.shape

        # queries => [B,H,L_Q,D]
        queries = queries.transpose(2,1)
        # keys => [B,H,L_K,D]
        keys = keys.transpose(2,1)
        # values => [B,H,L_V,D]
        values = values.transpose(2,1)

        # U_part = c*ln(L_k)
        U_part = self.factor*np.ceil(np.log(L_K)).astype('int').item()
        # u => c*ln(L_q)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        # 防止溢出
        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q

        scores_top, index = self._prob_QK(queries,keys,sample_k=U_part,n_top=u)

        # 添加scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # 得到context信息
        context = self._get_initial_context(values, L_Q)
        # 更新context信息
        context, attn = self._update_context(context,values,scores_top,index,L_Q,attnMask)
        # [B,H,L,E] => [B,L,H,E]
        return context.transpose(2,1).contiguous(), attn

class AttentionLayer(nn.Module):
    """

    """

    def __init__(self,attention,dModel,nHeads,dKeys=None,dValues=None,mix=False):
        """

        :param attention:
        :param dModel:
        :param nHeads:
        :param dKeys:
        :param dValues:
        :param mix: bool,表示
        """
        super(AttentionLayer, self).__init__()

        # 定义好
        dKeys = dKeys or (dModel//nHeads)
        dValues = dValues or (dModel//nHeads)

        # 做线性变换处理
        self.inner_attention = attention
        self.query_projection = nn.Linear(dModel, dKeys * nHeads)
        self.key_projection = nn.Linear(dModel, dKeys * nHeads)
        self.value_projection = nn.Linear(dModel, dValues * nHeads)
        self.out_projection = nn.Linear(dValues * nHeads, dModel)
        self.nHeads = nHeads
        self.mix = mix

    def forward(self,queries, keys, values, attnMask):
        """

        :param queries:
        :param keys:
        :param values:
        :param attnMask:
        :return: out => [B,L,512]
        """

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.nHeads

        # queries => [B,L,H,E]
        queries = self.query_projection(queries).view(B, L, H, -1)
        # keys => [B,S,H,E]
        keys = self.key_projection(keys).view(B, S, H, -1)
        # values => [B,S,H,E]
        values = self.value_projection(values).view(B, S, H, -1)

        # out => [B,H,n_top,E]
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attnMask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        # out => [B,L,512]
        out = out.view(B, L, -1)
        return self.out_projection(out), attn































