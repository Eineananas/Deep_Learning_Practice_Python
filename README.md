# Deep_Learning_Practice_Python


#词向量：50-300维，有两个策略
#CBOW模型：由上下文词预测中心词
#skip-gram模型：由中心词预测上下文词（更常用）
#负采样：输入一些不存在的组合，然后赋标记为0
#窗口大小：窗口数为2，则上下文各取两个
#glove是word2vec的拓展，是基于全局词频统计的此表示工具
#可以捕捉到全局语料的统计信息，综合了word2vec的局部上下文窗口，和LSA的全局矩阵分解（有点像LDA）
#LSA不擅长做词语类比（men->women=king->queen）,而word2vec擅长将上下文词把词语映射到向量空间中，进行类比
#LSA能够捕捉到全局语料的统计信息
#共现矩阵：对称矩阵，一来表达出两个词语共现的概率，二来表现出两个词语在共现时的距离
#共现矩阵的元素计算=共现次数*权重递减函数
#权重递减函数表现出两个词语在共现时的距离
#Glove将单词表达成实数组成的向量，可以捕捉到单词之间的一些语义特性，如相似性、类比性
#而且这些向量可以通过向量运算计算（如欧氏距离、余弦）出其相似度


#语言模型：计算一个句子是句子的概率
#前向语言模型：强调当前词语出现的概率对于前面出现词语的依赖关系
#还有后向语言模型
#双向语言模型（也就是ELMO，基于bi-LSTM,双向LSTM），就是同时包含前向和后向语言模型
#ELMO的缺点：运行速度
![image](https://github.com/Eineananas/Deep_Learning_Practice_Python/assets/133489015/b4d1957d-b6fb-4eaf-b976-9beb17939b4c)


#BERT, Bidirectional Encoder Representations from Transformers
#相对于RNN而言（按照顺序处理：会导致前后依赖，梯度消失/爆炸，处理速度受限）
#Transformer是序列转换模型，可以并行运算，速度快，能够捕捉上下文的信息
#序列转换模型：讲一个序列转化成另一个序列（Encoder*-*Decoder），如机器翻译
#attention提取上下文信息的方式：基于赋权重
![image](https://github.com/Eineananas/Deep_Learning_Practice_Python/assets/133489015/8df890c3-9785-4e9e-be0d-189654a27d08)
![image](https://github.com/Eineananas/Deep_Learning_Practice_Python/assets/133489015/982c809a-049d-441e-9c61-20ec8abc0c94)
![image](https://github.com/Eineananas/Deep_Learning_Practice_Python/assets/133489015/4724d396-0554-4abf-96c6-5b4b422c58c9)



