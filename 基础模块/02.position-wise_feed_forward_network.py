'''
前馈网络
重点是理解position-wise的含义，即对每个位置的特征进行相互独立的前馈计算，
也就是每个位置的token分别各自进行一次非线性的变换

逻辑比较简单，就是两个全连接层，中间用ReLU激活函数连接
输入形状：[batch_size, seq_len, d_model]
输出形状：[batch_size, seq_len, d_model]

因为原文其实没有关于 dropout模块的引入，这里注释掉
'''

import torch
import torch.nn as nn

class position_wise_feed_forward_network(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        '''
        d_model:输入的维度，也就是嵌入后的维度
        d_ff：两层全连接层之间的维度
        dropout:dropout比例
        '''
        super(position_wise_feed_forward_network,self).__init__()
        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)
       # self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.w1(x).relu())
        #dropout层
        #return self.dropout(self.w2(self.w1(x).relu()))

    