#词向量：50-300维，有两个策略
#CBOW模型：由上下文词预测中心词
#skip-gram模型：由中心词预测上下文词（更常用）
#负采样：输入一些不存在的组合，然后赋标记为0
#窗口大小：窗口数为2，则上下文各取两个
# glove是word2vec的拓展，是基于全局词频统计的此表示工具
# 可以捕捉到全局语料的统计信息，综合了word2vec的局部上下文窗口，和LSA的全局矩阵分解（有点像LDA）
#LSA不擅长做词语类比（men->women=king->queen）,而word2vec擅长将上下文词把词语映射到向量空间中，进行类比
#LSA能够捕捉到全局语料的统计信息
#共现矩阵：对称矩阵，一来表达出两个词语共现的概率，二来表现出两个词语在共现时的距离
# 共现矩阵的元素计算=共现次数*权重递减函数
# 权重递减函数表现出两个词语在共现时的距离
# Glove将单词表达成实数组成的向量，可以捕捉到单词之间的一些语义特性，如相似性、类比性
# 而且这些向量可以通过向量运算计算（如欧氏距离、余弦）出其相似度


#语言模型：计算一个句子是句子的概率
#前向语言模型：强调当前词语出现的概率对于前面出现词语的依赖关系
#还有后向语言模型
# 双向语言模型（也就是ELMO，基于bi-LSTM,双向LSTM），就是同时包含前向和后向语言模型
# ELMO的缺点：运行速度

# BERT, Bidirectional Encoder Representations from Transformers
# 相对于RNN而言（按照顺序处理：会导致前后依赖，梯度消失/爆炸，处理速度受限）
# Transformer是序列转换模型，可以并行运算，速度快，能够捕捉上下文的信息
# 序列转换模型：讲一个序列转化成另一个序列（Encoder*-*Decoder），如机器翻译
# attention提取上下文信息的方式：基于赋权重


from __future__ import print_function
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


glove_file = datapath('C:/Users/WeiTh/Desktop/glove/glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
print(model.most_similar('banana'))
result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))
#Stanford大学训练的一个Glove,你玩玩就行了

#接下来是我们自己训练的Glove


import argparse
import pprint
import gensim
from glove import glove
from glove import corpus
import pkuseg
#北大写的一个分词的包
pkuseg.test(r'不要等到毕业以后.txt', r'不要等到毕业以后分词.txt', model_name = "default", user_dict = "default", postag = False, nthread = 5)


# In[ ]:


#1.准备数据集
with open(r'不要等到毕业以后分词.txt','r',encoding='utf-8') as f:
    sentense = [line.replace('\n','').split(' ')  for line in f.readlines() if line.strip()!='']
#sentense = [['你','是','谁'],['我','是','中国人']]
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)#10
#corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
#Dict size: 2485
print('Collocations: %s' % corpus_model.matrix.nnz)
#Collocations: 71160

# 2.训练
glove = Glove(no_components=100, learning_rate=0.05)#no_components 维度，可以与word2vec一起使用。
glove.fit(corpus_model.matrix, epochs=10, no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

#3.corpus模型保存与加载
corpus_model.save('corpus.model')
corpus_model = Corpus.load('corpus.model')


# In[29]:


glove.dictionary


# In[30]:


# 指定词条词向量
glove.word_vectors[glove.dictionary['你']]


# In[31]:


# 相似词
glove.most_similar('专业', number = 10)


#英文
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = r"C:/Users/WeiTh/Desktop/ELMO/options.json" # 配置文件地址
weight_file = r"C:/Users/WeiTh/Desktop/ELMO/weights.hdf5" # 权重文件地址

# 这里的1表示产生一组线性加权的词向量。
# 如果改成2 即产生两组不同的线性加权的词向量。
elmo = Elmo(options_file, weight_file, 1, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentence_lists = [['I', 'love', 'you', '.'], ['Sorry', ',', 'I', 'don', "'t", 'love', 'you', '.']]
character_ids = batch_to_ids(sentence_lists)
print(character_ids.shape)
print(character_ids)
#首先会把单词处理一下，只是编了个号
#torch.Size([2, 8, 50])，两个句子，所以是2；最多的句子有8个单词，所以其他句子补齐为8；
#固定是50个字符，因为一个单词基本上只有50个字母
#然后按照这个编号，再按照语境，生成词向量
embeddings = elmo(character_ids)['elmo_representations'][0]
#elmo_mask = elmo(character_ids)['mask']
print(embeddings.shape)
#torch.Size([2, 8, 1024])，两个句子，所以是2；最多的句子有8个单词，所以其他句子补齐为8；
#词向量长度为1024
print(embeddings)
#最终输出的词向量

#中文
#哈工大的一个中文生成词向量的包
from elmoformanylangs import Embedder
e = Embedder('C:/Users/WeiTh/Desktop/ELMO/zhs.model/')  #绝对路径

sents = [['今', '天', '天氣', '真', '好', '阿'],
['潮水', '退', '了', '就', '知道', '誰', '沒', '穿', '褲子']]
# the list of lists which store the sentences
# after segment if necessary.

ch_em = e.sents2elmo(sents,output_layer=-1)
# will return a list of numpy arrays
# each with the shape=(seq_len, embedding_size)

# output_layer: the target layer to output.
# 0 for the word encoder
# 1 for the first LSTM hidden layer
# 2 for the second LSTM hidden layer
# -1 for an average of 3 layers. (default)
# -2 for all 3 layers

print(ch_em[0].shape)






