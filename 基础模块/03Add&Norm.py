'''
先分别进行残差连接和归一化的代码
再整合起来
'''

'''
01.残差连接
一种跳跃连接

需要对sublayer的结果进行dropout,在残差连接前应用在子层输出上，防止子层过拟合
然后与输入x进行残差连接
初始化：输入dropput
前向传播：输入子层输入x和子层函数sublayer
'''
import torch
import torch.nn as nn

class residual_connection(nn.Module):
    def __init__(self,dropout=0.1):
        '''
        dropout:dropout的概率
        '''
        super(residual_connection,self).__init__()
        self.dropout=nn.Dropout(p=dropout)
    
    def forward(self,x,sublayer):
        '''
        前向传播
        输入：子层的输入+子层函数
        输出：残差连接后的结果
        '''
        return x+self.dropout(sublayer(x))


'''
02.层归一化

1.层归一化和批量归一化的区别
层归一化：对每个样本的所有特征维度进行归一化，是在最后一个维度进行的
批量归一化：对每个特征维度的所有样本进行归一化，是在第一个维度进行的
2.层归一化的公式中有两个可学习的参数
γ：缩放因子，默认初始化为1
β：偏移因子，默认初始化为0
epsilon：是一个非常小的常数，防止公式的分母为0
3.区别于线性层的变换功能：
γ是一个一维向量，进行的是逐个特征的对应相乘，而线性层执行的是对所有特征维度的线性变换是matmul操作



初始化：输入模型维度和epsilon
前向传播：残差连接后的结果
'''

class LayerNorm(nn.Module):
    def __init__(self,d_model,epsilon=1e-6):
        '''
        参数说明：
        d_model：特征维度，在这里是嵌入后的维度
        epsilon：是一个非常小的常数，防止公式的分母为0
        '''
        self.d_model=d_model
        self.epsilon=epsilon
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
    
    def forward(self,x):
        '''
        前向传播
        输入：子层的输入
        输出：归一化后的输出
        '''
        mean=x.mean(dim=-1,keepdim=True)#keepdim=True保持维度不变，否则会减少一个维度
        std=x.std(dim=-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.epsilon)+self.beta
    
'''
03 Add&Norm
公式为：Output=LayerNorm(x+SubLayer(x))
post-norm模式 一共三个流程：
1.先对子层的结果进行dropout
2.再进行残差连接
3.再对加和结果进行归一化
其中sublayer就是子层，子层可以是注意力层和前馈网络层这些网络层

pre-norm模式:先归一化再drop最后连接
'''

class SublayerConnection(nn.Module):
    def __init__(self,d_model,dropout=0.1,epsilon=1e-6):
        super(SublayerConnection,self).__init__()
        self.add=residual_connection(dropout)
        self.norm=LayerNorm(d_model,epsilon)
    
    def forward(self,x,sublayer):
        return self.norm(self.add(x,sublayer(x)))

#第二种方式：还是调用norm模块但是实现残差模块
#class SublayerConnection(nn.Module):
#    def __init__(self,d_model,dropout=0.1,epsilon=1e-6):
#        super(SublayerConnection,self).__init__()
#        self.norm=LayerNorm(d_model,epsilon)
#        self.dropout=nn.Dropout(p=dropout)
    
#    def forward(self,x,sublayer):
#        return self.norm(x+self.dropout(sublayer(x)))
