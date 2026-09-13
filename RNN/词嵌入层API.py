"""

词嵌入层：
    把离散的词映射成稠密向量的层，即把词(或者词对应的索引)转成词向量

"""
import torch
import jieba
import torch.nn as nn


text = "北京欢迎你，为你开天辟地！"
#使用jieba分词器分词
words = jieba.lcut(text)
print(f'分词结果：{words}')

#创建词嵌入层
#参1：词表大小(词的个数)，参2：词向量的维度
embed = nn.Embedding(len(words),4)

#enumerate():返回列表中每个值及其对应的索引
for i,word in enumerate(words):
    #把词索引(张量形式)转换成词向量
    word_vector = embed(torch.tensor(i))
    print(f'词：{word}，\t\t\t词向量:{word_vector}')
