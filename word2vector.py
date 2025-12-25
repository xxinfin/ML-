#-*-coding:UTF-8-*-

import os
import argparse
import urllib.request
import zipfile
import collections
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn
from tempfile import gettempdir
from tqdm import tqdm
import scipy.spatial.distance

from torch.utils.tensorboard import SummaryWriter

current_path=os.getcwd()
parser=argparse.ArgumentParser()
parser.add_argument(
    "--log_dir",
    type=str,
    default=os.path.join(current_path, "log"),
    help="The log directory for TensorBoard summaries."
)
FLAGS,unpared=parser.parse_known_args()

if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
url="http://mattmahoney.net/dc/"

def maybe_download(filename,expected_bytes):
    """

    下载数据集文件并且校验长度
    :param filename: 文件名称
    :param expected_bytes: 文件长度
    :return: 文件句柄
    """
    local_filename=os.path.join("E:\\",filename)
    if not os.path.exists(local_filename):
        local_filename,_=urllib.request.urlretrieve(url+filename,local_filename)
    statinfo=os.stat(local_filename)
    if statinfo.st_size==expected_bytes:
        print("下载完成")
    else:
        print(statinfo.st_size)
        raise Exception("无法下载"+local_filename+"请尝试在浏览器下载")
    return local_filename
filename=maybe_download("text8.zip",31344016)
print(filename)



def read_data(filename):
    """
    将压缩包解压为词表
    :param filename: 压缩包文件庐江
    :return: 词表数据
    """
    with zipfile.ZipFile(filename) as f:
        data=f.read(f.namelist()[0]).decode(encoding="utf-8").split()
    return data
vocabulary=read_data(filename)
print("Data size",len(vocabulary))


vocabulary_size=50000#词表大小
feature_length=100 #特征长度
C=3 #窗口大小
K=100 #负样本大小
epoch=1 #迭代次数
batch_size=128 #取样大小


def build_dataset(words,n_words):
    """
    创建数据集
    :param words: 词表
    :param n_words: 关注词的数量
    :return:
    """
    count=[["UNK",-1]]
    count.extend(collections.Counter(words).most_common(n_words-1))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    for word in words:
        index=dictionary.get(word,0)
        if index==0:
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reversed_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reversed_dictionary


data,counts,dictionary,reversed_dictionary=build_dataset(vocabulary,vocabulary_size)
del vocabulary
print("Most common wprds(+UNK)",counts[:5])
print("Sample data",data[:10],[reversed_dictionary[i] for i in data[:10]])



class WordEmbeddingDataset(Dataset):
    def __init__(self,data,count,dictionary,reversed_dictionary):
        self.data = torch.Tensor(data).long()
        word_counts = np.array([count[1] for count in counts], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)
        self.word_freqs = torch.Tensor(word_freqs / np.sum(word_freqs))
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        center_word = self.data[item]
        pos_indics = list(range(item - C, item)) + list(range(item + 1, item + C + 1))
        pos_indics = [i % len(self.data) for i in pos_indics]
        pos_words = self.data[pos_indics]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(data, counts, dictionary, reversed_dictionary)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



class EmbeddingModel(nn.Module):
    def __init__(self,vocabulary_size,feature_length):
        super(EmbeddingModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.feature_length = feature_length

        initrange = 0.5 / self.feature_length
        self.out_embed = nn.Embedding(self.vocabulary_size, self.feature_length, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocabulary_size, self.feature_length, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.out_embed(pos_labels)
        neg_embedding = self.out_embed(neg_labels)

        # 计算损失
        input_embedding = input_embedding.unsqueeze(2)
        log_pos = torch.bmm(pos_embedding, input_embedding).squeeze()
        log_neg = torch.bmm(neg_embedding, input_embedding).squeeze()

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


model=EmbeddingModel(vocabulary_size,feature_length)
if torch.cuda.is_available():
    model=model.cuda()


optimizer=torch.optim.SGD(model.parameters(),lr=0.2)
writer=SummaryWriter("log")
for e in range(epoch):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(tqdm(dataloader, desc=str(e))):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        if torch.cuda.is_available():
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = torch.mean(model(input_labels, pos_labels, neg_labels))
        writer.add_scalar("test_loss", loss, e * len(dataloader) + i)
        loss.backward()
        optimizer.step()



embedding_weights=model.input_emdeddings()
np.save("feature_length-{}".format(feature_length),embedding_weights)
torch.save(model.state_dict(),"embedding-{}.th".format(feature_length))


model.load_state_dict(torch.load("embedding-{].th".format(feature_length)))
embedding_weights=model.input_emdeddings()



def find_nearest(word):
    index = dictionary[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [reversed_dictionary[i] for i in cos_dis.argsort()[:10]]


for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
    print(word, find_nearest(word))

man_idx = dictionary["man"]
king_idx = dictionary["king"]
woman_idx = dictionary["woman"]
embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
for i in cos_dis.argsort()[:20]:
    print(reversed_dictionary[i])


meta_list=list(dictionary.keys())
writer.add_embedding(embedding,metadata=meta_list)
writer.close()