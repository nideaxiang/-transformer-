'''
mask模块
01 填充mask 在编码器/解码器中均存在
02 未来mask 只在解码器中存在

03 将两者混合起来
'''
import torch

def create_padding_mask(seq,padding_idx=0):
    """
    参数：
        seq: 输入序列，形状为 [batch_size, seq_len]
        padding_idx: 填充索引，默认值为 0
    返回：
        mask: 填充掩码，形状为 [batch_size, 1, 1, seq_len]
    """
    mask=(seq != padding_idx).unsqueeze(1).unsqueeze(2) #元组，变化后形状为[batch_size,1,1,seq_len]
    return mask


def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size)).type(torch.bool)  # 下三角矩阵
    return mask  # (seq_len, seq_len)

a=create_look_ahead_mask(5)
print(a)

def create_decoder_mask(tgt,pad_idx=0):
    """
    参数：
        tgt: 目标序列，形状为 [batch_size, tgt_seq_len]
    返回：
        mask: 解码器掩码，形状为 [batch_size, 1, tgt_seq_len, tgt_seq_len]
    """
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)  # 形状为 [batch_size, 1, 1, tgt_seq_len]
    tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device) #形状为[tgt_seq_len,tgt_seq_len]

    combined_mask = tgt_padding_mask & tgt_look_ahead_mask.unsqueeze(0)  # 形状为 [batch_size, 1, tgt_seq_len, tgt_seq_len]
    return combined_mask

tgt_seq = torch.tensor([[1, 2, 3, 4, 0]])  # 0 表示 <PAD>
print(create_decoder_mask(tgt_seq))
