'''
嵌入模块
输入：[batch_size,seq_len]
输出：下游的token embedding序列，[batch_size,seq_len,d_model]

流程：
1）查表嵌入
2）缩放：乘以sqrt(d_model)

1.为什么进行了分词还需要嵌入？
使得token之间能够有高维语义联系

'''
import math
import torch
import torch.nn as nn
class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        '''
        参数说明：
        vocab_size：词汇表大小
        d_model：嵌入维度
        '''
        super(Embeddings,self).__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.scale_factor=math.sqrt(d_model)
    def forward(self,x):
        '''
        前向传播
        输入：token索引序列，[batch_size,seq_len]
        输出：token embedding序列，[batch_size,seq_len,d_model]
        '''
        return self.embedding(x)*self.scale_factor

#可以看看embedding矩阵的样子，随机初始化一个嵌入层，词汇表大小为10000，嵌入维度为512
emb = Embeddings(10000, 512)
#此时矩阵是随机初始化的，每个元素都是从标准正态分布中采样的，还不具有语义信息

print("Embedding matrix shape:", emb.embedding.weight.shape)
print("Embedding matrix example row:", emb.embedding.weight[0][:10])


if __name__ == '__main__':
    # 测试嵌入模块
    vocab_size = 10000
    d_model = 512
    emb = Embeddings(vocab_size, d_model)
    print(emb.embedding.weight.shape)  # 应该输出 [10000, 512]