'''
05位置编码模块

1.首先是位置编码的必要性
我们为的是能具有自然语言的顺序信息，因为自然语言的顺序带有重要的语义信息。

2.编码的公式采用正弦余弦函数
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

3.我们会定义一个最大的位置编码长度max_len，但实际使用时不一定全部都取到，
所以在forward中会根据实际的位置编码长度来截取。

4.思路：初始化位置矩阵创建位置索引，计算位置编码，将位置编码注册为缓冲区
'''
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000,dropout=0.1):
        '''
        max_len: 位置编码的最大长度，适应不同长度的输入序列
        '''
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #初始化位置编码矩阵，形状是[max_len,d_model]
        pe=torch.zeros(max_len,d_model)
        #创建位置索引,size是[max_len,1]
        position=torch.arange(0,max_len).unsqueeze(1)
        #计算分母部分，也就是频数
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        #结合频数与位置,sin函数使用偶数索引,cos函数使用奇数索引 ::代表步长为2
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        #将pe的维度加上batch_size,形状是[1,max_len,d_model]
        pe = pe.unsqueeze(0)
        #pe是固定的，不参与训练，所以注册为缓冲区
        self.register_buffer('pe',pe)

    def forward(self,x):
        '''
        输入：
        x: 输入序列的位置编码，形状是[batch_size,seq_len,d_model]
        输出：
        x: 输入序列的位置编码，形状是[batch_size,seq_len,d_model]
        '''
        #取出相同长度的位置编码加到嵌入上
        x=x+self.pe[:,:x.size(1),:]
        return self.dropout(x)

if __name__ == '__main__':
    # 测试位置编码模块
    d_model = 512
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    print(pe.pe.shape)  # 应该输出 [1, 100, 512]
