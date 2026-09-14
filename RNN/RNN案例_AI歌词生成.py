"""

实现步骤：
    1. 获取数据，进行分词，获取词表
    2. 数据预处理，构建数据集
    3. 搭建RNN神经网络
    4. 训练模型
    5. 模型预测

"""

import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 获取数据，进行分词，获取词表
def build_vocab():
    #1. 定义变量，记录：去重后的所有的词，每行文本分词结果
    unique_words,all_words = [],[]
    #2. 遍历数据集，获取到每行文本
    for line in open('./data/jaychou_lyrics.txt','r',encoding='utf-8'):
        # 获取到每行歌词，进行分词
        words = jieba.lcut(line)            #一个列表
        # 所有分词结果记录到all_words中
        all_words.append(words)             #从一维变成了二维
        # 遍历分词结果，去重后，添加到unique_words中
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    #3. 统计语料中去重后词的数量
    word_count = len(unique_words)
    # 4. 构建词表，字典形式，key是词，value是词索引
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    # 确保空格在词表里，否则下面会 KeyError
    if ' ' not in word_to_index:
        word_to_index[' '] = len(word_to_index)

    # 5. 歌词文本用词表索引表示（展平成一维长列表）
    corpus_idx = []
    for words in all_words:
        for word in words:
            corpus_idx.append(word_to_index[word])
        corpus_idx.append(word_to_index[' '])  # 每行末尾加空格作分隔

    # 6. 返回
    return unique_words, word_to_index, word_count, corpus_idx


# 2. 数据预处理，构建数据集
# 定义数据集类，继承Dataset类
class LyricsDataset(torch.utils.data.Dataset):
    #1. 初始化词索引，词个数
    def __init__(self, corpus_idx, num_chars):
        # 文档数据中词的索引（一维长列表）
        self.corpus_idx = corpus_idx
        # 每个句子中词的个数（滑动窗口长度）
        self.num_chars = num_chars
        # 样本数量 = 总词数 - 窗口长度
        # 减 num_chars 是为了保证取 y 时 start+1 : end+1 不越界
        self.number = len(corpus_idx) - num_chars

    #2. 当使用len(obj)时，自动调用此方法
    def __len__(self):
        # 返回样本数量
        return self.number

    #3. 当使用obj[index]时，自动调用此方法
    def __getitem__(self, idx):
        # idx 由 DataLoader 根据 __len__ 生成，一定在 [0, number) 范围内，不会越界
        # 当前样本的起始索引
        start = idx
        # 当前样本的结束索引（取 num_chars 个词）
        end = start + self.num_chars
        # 输入值：从文档中取出 start ~ end 的词索引 -> 作为 x（前 num_chars 个词）
        x = self.corpus_idx[start:end]
        # 输出值：后移一位，即 start+1 ~ end+1 -> 作为 y（预测的下一个词序列）
        y = self.corpus_idx[start + 1:end + 1]
        # 返回输入值和输出值 -> 张量形式
        return torch.tensor(x), torch.tensor(y)


# 3. 搭建RNN神经网络
class TextGenerator(nn.Module):
    #1. 初始化方法
    def __init__(self,word_count):
        super().__init__()
        # 初始化词嵌入层：语料中词的数量，词向量维度
        self.embd = nn.Embedding(word_count, 128)
        # 循环网络层：词向量维度，隐藏层维度，网络层
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层(全连接层)：特征向量维度(和隐藏层维度保持一致)，词表中词的个数
        self.output = nn.Linear(256, word_count)

    #2. 前向传播方法
    def forward(self, inputs, hidden):
        # 初始化词嵌入层处理
        embd = self.embd(inputs)
        # rnn处理
        rnn_output, hidden = self.rnn(embd.transpose(0, 1), hidden)
        # 全连接，输入内容必须是二维数据，即词的数量*词的维度
        output = self.output(rnn_output.reshape(shape=(-1, rnn_output.shape[-1])))
        # 返回预测结果和隐藏层结果
        return output, hidden

    #3. 隐藏层初始化方法
    def init_hidden(self,bs):
        # 隐藏层初始化：[网络层数，batch，隐藏层维度]
        return torch.zeros(1, bs, 256,device=next(self.parameters()).device)


# 4. 训练模型
def train():
    #1. 创建词典
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    #2. 获取数据集
    lyrics = LyricsDataset(corpus_idx, num_chars=50)
    #3. 初始化模型
    model = TextGenerator(word_count).to(device)
    #4. 创建数据加载器对象,每批次64个句子，每个句子50个词
    lyrics_dataloader = DataLoader(lyrics, batch_size=64, shuffle=True)
    #5. 定义损失函数，定义优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #6. 训练模型
    epochs = 35
    for epoch in range(epochs):
        # 定义本轮开始训练时间，批次数，训练总损失
        start,iter_num,total_loss = time.time(),0,0
        for x,y in lyrics_dataloader:
            x, y = x.to(device), y.to(device)
            # 获取隐藏层初始值
            hidden = model.init_hidden(x.size(0))
            # 模型计算
            output, hidden = model(x, hidden)
            # 计算损失
            y = torch.transpose(y, 0, 1).reshape(shape=(-1, ))
            loss = criterion(output, y)
            # 梯度清零 + 反向传播 + 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iter_num += 1
        print(f'epoch:{epoch+1}, time:{time.time()-start:.3f}s, loss:{total_loss/iter_num:.4f}')

    #7. 保存模型
    torch.save(model.state_dict(), './model/rnn_model.pth')


# 5. 模型预测
def evaluate(start_word,sentence_length):
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    model = TextGenerator(word_count).to(device)
    model.load_state_dict(torch.load('./model/rnn_model.pth', map_location=device))
    model.eval()
    hidden = model.init_hidden(1)
    if start_word not in word_to_index:  # 检查起始词
        print(f'"{start_word}" 不在词表中')
        return
    # 将输入的开始词转换成索引
    word_idx = word_to_index[start_word]
    # 定义列表，存放：产生的词的索引
    generate_sentence = [word_idx]
    # 遍历句子长度，获取到每个词
    with torch.no_grad():                                   #不计算梯度
        for i in range(sentence_length):
            input_tensor = torch.tensor([[word_idx]]).to(device)
            output, hidden = model(input_tensor, hidden)
            word_idx = torch.argmax(output, dim=-1).item()
            generate_sentence.append(word_idx)

    # 将索引转成词，并打印
    for idx in generate_sentence:
        print(unique_words[idx], end=' ')



if __name__ == '__main__':
    # unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # # print(f'词的数量:{word_count}')
    # # print(f'去重后的词:{unique_words}')
    # # print(f'每个词的索引:{word_to_index}')
    # # print(f'文档中每个词对应的索引:{corpus_idx}')

    # # dataset = LyricsDataset(corpus_idx, num_chars=5)
    # # print(f'句子数量:{len(dataset)}')
    # # x,y = dataset[0]
    # # print(f'输入值：{x}')
    # # print(f'输出值：{y}')

    # model = TextGenerator(word_count)

    # train()

    evaluate(start_word='青色', sentence_length=30)