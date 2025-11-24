"""
01scaled_dot_attention部分
最原始的注意力计算
输入 变换好的 Q K V mask
输出 注意力表征上下文 以及 注意力权重

注意mask的位置，是在传入softmax前
"""


from turtle import forward
import torch
import torch.nn.functional as F
import math
import torch.nn as nn

def attention(Q,K,V,mask=None):
    """
    参数：
    Q: 查询矩阵，形状为(batch_size, seq_len, embed_dim)
    K: 键矩阵，形状为(batch_size, seq_len, embed_dim)
    V: 值矩阵，形状为(batch_size, seq_len, embed_dim)
    mask: 可选的掩码矩阵，形状为(batch_size, seq_len, seq_len)
    """
    embed_dim = Q.size(-1)
    #计算KV点积，并进行缩放，得到注意力分数
    attention_score=torch.matmual(Q,K.transpose(-2,-1))/math.sqrt(embed_dim)
       #transpose(-2,-1)表示将K的最后两个维度交换，得到K的转置矩阵

#掩码处理：若存在掩码mask,则将注意力分数中对应位置的分数设置为- inf
    # 这样在后续的softmax归一化中，这些位置的权重将接近于0，从而实现了对序列中某些位置的注意力屏蔽。
    if mask is not None:
        attention_score=attention_score.masked_fill(mask==0,float('-inf'))

#对每个查询与键的结果进行softmax归一化
    attention_weight=F.softmax(attention_score,dim=-1) 
        #dim=-1表示对最后一个维度进行softmax归一化,也就是在行上进行归一化


#加权注意力分数，获得上下文向量
    attenton_context=torch.matmul(attention_weight,V)


    return attenton_context,attention_weight

#-------------------------------------------------------------------------------------



'''
02考虑 KQV的来源的单头注意力机制
文本序列——>tokenize——>embedding——>KQV
K Q V来源于embedding以后的同一个词表征进行的线性变化
线性变换生成K Q V，
然后再执行缩放点积注意力机制

nn.Linear()线性层
torch.nn.Linear(in_features, # 输入的神经元个数
           out_features, # 输出神经元个数
           bias=True # 是否包含偏置
           )
输入qkv或者说某一序列x，qkv三者的来源可能相同，可能不同

'''



class singlehead_attention(nn.Module):
    def __init__(self,embed_dim):
        """
        单头注意力机制
        参数：
        embed_dim: 输入的词向量embedding的维度
        """
        super().__init__()
        self.embed_dim=embed_dim
        self.w_q=nn.Linear(embed_dim,embed_dim)
        self.w_k=nn.Linear(embed_dim,embed_dim)
        self.w_v=nn.Linear(embed_dim,embed_dim)

    def forward(self,q,k,v,mask=None):
        """
        前向传播
        参数：
        q: 查询矩阵，形状为(batch_size, seq_len_q, embed_dim)
        k: 键矩阵，形状为(batch_size, seq_len_k, embed_dim)
        v: 值矩阵，形状为(batch_size, seq_len_v, embed_dim)
        mask: 可选的掩码矩阵，形状为(batch_size, seq_len_q, seq_len_k)
        """
        # 线性变换
        Q=self.w_q(q)
        K=self.w_k(k)
        V=self.w_v(v)
        #得到了变换后的QKV可以进行点积缩放归一化
        attenton_context,attention_weight=attention(Q,K,V,mask)
        # 输出注意力表征上下文，形状为(batch_size, seq_len_q, embed_dim)
        return attenton_context,attention_weight

#-------------------------------------------------------------------------------------
    
"""
03 单头自注意力的计算 q k v 来源于同一个序列x
则在前向传播时候只需要要传入一个相同的序列x
但基本的计算流程都在singlehead_attention中实现了，可以直接调用
1.从序列x得到qkv 
2.进行注意力计算

"""

class self_attention(nn.Module):
    def __init__(self,embed_dim):
        """
        单头自注意力机制
        参数：
        embed_dim: 输入的词向量embedding的维度
        """
        super(self_attention,self).__init__()
        self.attention=singlehead_attention(embed_dim) #调用定义好的单头注意力机制

    def forward(self,x,mask=None):
        """
        前向传播
        参数：
        x: 输入序列，形状为(batch_size, seq_len, embed_dim)
        mask: 可选的掩码矩阵，形状为(batch_size, seq_len, seq_len)
        """
        # 调用单头注意力机制进行计算
        #q = k= v
        attenton_context,attention_weight=self.attention(x,x,x,mask)
        # 输出注意力表征上下文，形状为(batch_size, seq_len_q, embed_dim)
        return attenton_context,attention_weight



#-------------------------------------------------------------------------------------
'''
04 交叉注意力机制
每一次的查询来自于encoder,而键值来源于decoder
encoder和decoder 自身先要执行一次self_attention,
得到encoder和decoder的注意力表征上下文
然后再执行交叉注意力机制
交叉注意力的本质是利用已经生成的内容计算其在原句子中相关性，
从而获得整个句子的上下文信息，基于这些信息生成下一个token


以机器翻译中的中译英任务为例：对于中文句子“中国的首都是北京”，
假设模型的编码器已经生成了部分译文“The capital of China is”，此时需要预测下一个单词。
在这一阶段，解码器输出右移动为输入，计算一次自注意力，再基于交叉注意力机制，使用当前已生成的译文“The capital of China is”
的编码表示作为查询，并将编码器对输入句子“中国的首都是北京”编码表示作为键和值，
通过计算查询与键之间的匹配程度，生成相应的注意力权重，
以此从值中提取上下文信息，基于这些信息生成下一个可能的单词（token），比如：“Beijing”。

'''
class cross_attention(nn.Module):
    def __init__(self,embed_dim):
        """
        交叉注意力机制
        参数：
        embed_dim: 输入的词向量embedding的维度
        """
        super(cross_attention,self).__init__()
        self.attention=singlehead_attention(embed_dim)
    
    def forward(self,encoder_output,decoder_output,mask=None):
        """
        前向传播
        参数：
        encoder_output: 编码器输出序列，形状为(batch_size, seq_len_encoder, embed_dim)
        decoder_output: 解码器输出序列，形状为(batch_size, seq_len_decoder, embed_dim)
        mask: 可选的掩码矩阵，形状为(batch_size, seq_len_decoder, seq_len_encoder)
        """
        # 调用单头注意力机制进行计算
        q = decoder_output
        k= encoder_output
        v= encoder_output
        attenton_context,attention_weight=self.singlehead_attention(q,k,v,mask)
        # 输出注意力表征上下文，形状为(batch_size, seq_len_decoder, embed_dim)
        return attenton_context,attention_weight


#-------------------------------------------------------------------------------------
'''
05 多头注意力机制
多头注意力机制是将多个单头注意力机制并行计算，
每个单头注意力机制关注输入序列的不同部分，
从而获得更丰富的上下文表示。

'''
class multihead_attention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        """
        多头注意力机制
        参数：
        embed_dim: 输入的词向量embedding的维度
        num_heads: 头数

        注意 最后进行拼接contact时候，按照的是最后一维进行拼接,因为切割时候
        """
        super(multihead_attention,self).__init__()
        self.num_heads=num_heads
        self.embed_dim_for_head=embed_dim//num_heads
        
        #每个头都需要定义QKV
        self.w_q=nn.ModuleList([nn.Linear(embed_dim,self.embed_dim_for_head) for _ in range(num_heads)])
        self.w_k=nn.ModuleList([nn.Linear(embed_dim,self.embed_dim_for_head) for _ in range(num_heads)])
        self.w_v=nn.ModuleList([nn.Linear(embed_dim,self.embed_dim_for_head) for _ in range(num_heads)])
        self.w_o=nn.Linear(embed_dim,embed_dim)

    def forward(self,q,k,v,mask=None):
        """
       
        """
        # 计算每个头的注意力表征上下文
        head_outputs=[]
        for head in range(self.num_heads):
            # 计算当前头的QKV
            Q=self.w_q[head](q)
            K=self.w_k[head](k)
            V=self.w_v[head](v)
            # 计算当前头的注意力表征上下文
            attenton_context_head,attention_weight_head=self.attention(Q,K,V,mask=None)
            head_outputs.append(attenton_context_head)
        # 拼接所有头的输出
        attenton_context=torch.cat(head_outputs,dim=-1) #按照最后一个维度拼接
        # 线性变换得到最终的注意力表征上下文
        mul_attenton_context=self.w_o(attenton_context)
        return mul_attenton_context

#-------------------------------------------------------------------------------------
f'''
06优化细节

1.优化多头注意力循环
激活部分不再进行modulelist的生成，
放弃循环遍历每个头来计算注意力，而是采用一次性计算Q K V然后拆分为多头的形式（也就是拆分操作）
先计算大的qkv,再把一个大 Q K V 矩阵中属于不同 head 的参数块分割开来计算注意力

拆分的实现在forward（）中，使用重塑和转置
1)拆分：把大的[batch_size, seq_len, embed_dim] 拆分为 [batch_size, seq_len, num_heads, embed_dim_for_head]
2)转置：还需要进行transpose维度交换，把seq_len换到倒数第二个维度，交换num_heads到第二个维度，而embed_dim_for_head保持放在最后一个维度
    转置的目的是：
    1. 为了后续的注意力点积的矩阵乘法，需要把 seq_len 放到倒数第二个维度，
        也就是说维度要满足(seq_len,head_dim)⋅(head_dim,seq_len)=(seq_len,seq_len)
    2. GPU 并行角度：在处理大批矩阵乘法时，需要维度尽可能为(batches, n, m)
    3. 为了后续的线性变换，需要把 embed_dim_for_head 保持放在最后一个维度。
方案：
    1)reshape+transpose
    2)直接使用view（使用view需要注意的是，view只能在连续的内存上操作，所以需要拼接时候使用contiguous()）

然后在最后一个维度上进行拼接，得到 [batch_size, seq_len, num_heads, embed_dim_for_head]
最后在 seq_len 维度上进行拼接，得到 [batch_size, seq_len, embed_dim]

2.点积缩放需要进行参数上说明的改变，因为点积计算是对后面两个维度进行的操作，我们在上一步进行了正确的维度处理
    参数:
        Q: 查询矩阵 (batch_size, num_heads, seq_len_q, head_dim)
        K: 键矩阵 (batch_size, num_heads, seq_len_k, head_dim)
        V: 值矩阵 (batch_size, num_heads, seq_len_v, head_dim)
        mask: 掩码矩阵 (1, 1, seq_len_q, seq_len_k) 或 (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, num_heads, seq_len_q, seq_len_k)

    返回:
        output: 注意力加权后的输出矩阵
        attention_weights: 注意力权重矩阵
3.按照论文的变量名称：
序列的嵌入维度：embed_dim 写为d_model
头数：num_heads写为h
每个头的嵌入维度： embed_dim_for_head 写为d_k

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h):
        '''
        d_model:词嵌入的维度
        h:注意力的头数
        '''
        super(MultiHeadAttention).__init__()
        assert d_model%h ==0 #必须能被整除
        self.d_model=d_model
        self.h=h
        
        #非共享的qkv线性转化层
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        
        #输出线性层
        self.w_o=nn.Linear(d_model,d_model)
    
    def forward(self,q,k,v,mask=None):
        '''
        q:查询序列(batch_size,seq_len_q,d_model)
        k:键序列(batch_size,seq_len_k,d_model)
        v:值序列(batch_size,seq_len_k,d_model)
        mask:可选的掩码矩阵(batch_size,1,seq_len_q,seq_len_k)或(1,1,seq_len_q,seq_len_k)
        '''
        batch_size=q.size(0)
        #如果是是交叉注意力q与kv的序列长度可能会不同
        seq_len_q=q.size(1)
        seq_len_k=k.size(1)
        #拆分+转置
        Q=self.w_q(q).view(batch_size,seq_len_q,self.h,-1).transpose(1,2)
        K=self.w_k(k).view(batch_size,seq_len_k,self.h,-1).transpose(1,2)
        V=self.w_v(v).view(batch_size,seq_len_k,self.h,-1).transpose(1,2)

        scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)
        concat_attention=scaled_attention.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        out=self.w_o(concat_attention)
        return out

    def scaled_dot_product_attention(self,Q,K,V,mask=None):
        '''
        Q: 查询矩阵 (batch_size, num_heads, seq_len_q, head_dim)
        K: 键矩阵 (batch_size, num_heads, seq_len_k, head_dim)
        V: 值矩阵 (batch_size, num_heads, seq_len_v, head_dim)
        mask: 掩码矩阵 (1, 1, seq_len_q, seq_len_k) 或 (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, num_heads, seq_len_q, seq_len_k)
        '''
        d_k = Q.size(-1) 
        scores=torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
