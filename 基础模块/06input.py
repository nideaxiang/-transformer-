'''
输入模块将分为编码器输入和解码器输入
编码器输入在机器翻译领域又称为Source Embedding
解码器输入在机器翻译领域又称为Target Embedding

注意点：编码器输出嵌入需要右移操作才能与位置嵌入拼接
但我们在代码中没有这个体现

'''

import torch
import torch.nn as nn
from 05Positionembedding import PositionalEncoding

from 04embedding import Embeddings

class SourceEmbedding(nn.Module):
    def __init__(self,src_vocab_size,d_model,max_len=5000,dropout=0.1):
        super(SourceEmbedding,self).__init__()
        self.embedding=Embeddings(src_vocab_size,d_model)   #形状从[batch_size,seq_len]到[batch_size,seq_len,d_model]
        self.positional_encoding=PositionalEncoding(d_model,max_len,dropout) 
    def forward(self,x):
        return self.positional_encoding(self.embedding(x))



class TargetEmbedding(nn.Module):
    def __init__(self,tgt_vocab_size,d_model,max_len=5000,dropout=0.1):
        super(TargetEmbedding,self).__init__()
        self.embedding=Embeddings(tgt_vocab_size,d_model)   #形状从[batch_size,seq_len]到[batch_size,seq_len,d_model]
        self.positional_encoding=PositionalEncoding(d_model,max_len,dropout) 
    def forward(self,x):
        return self.positional_encoding(self.embedding(x))

