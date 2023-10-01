# Word Vectors: 50-300 dimensions, two strategies
# CBOW Model: Predicts center word from surrounding words
# Skip-gram Model: Predicts surrounding words from center word (more common)
# Negative Sampling: Introduces some non-existing combinations, marked as 0
# Window Size: If window size is 2, it considers two words on either side as context words
# GloVe captures global corpus statistics, combining local context windows like word2vec and global matrix factorization like LSA (a bit like LDA)
# LSA isn't good at word analogies (e.g., 'man' is to 'woman' as 'king' is to 'queen'), while word2vec maps words to vector spaces and performs analogies
# LSA captures global corpus statistics
# Co-occurrence matrix: Symmetric matrix, representing the probability of two words occurring together and the distance between them when they co-occur
# Elements of co-occurrence matrix calculated as co-occurrence count * weight decay function
import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib

filePath = r'D:/PyCharm2022/pythonProject2/venv/data/sanguo.txt'
f = open(filePath, 'r', encoding='utf-8') # Read text
lines = []
for line in f: # Tokenize each line
    temp = jieba.lcut(line)  # Jieba word segmentation in precise mode
    words = []
    for i in temp:
        # Filter out all punctuation marks
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print(lines[0:5])  # Preview the tokenized results of the first 5 lines

# Call Word2Vec for training
# Parameters: size - word vector dimension, window - context window width, min_count - minimum word frequency for consideration
model = Word2Vec(lines, vector_size=20, window=2, min_count=3, epochs=7, negative=10, sg=1)
print("Word Vector for '孔明':\n", model.wv.get_vector('孔明'))
print("\nTop 20 words most similar to '孔明':")
print(model.wv.most_similar('孙权', topn=20))  # Top 20 words most similar to '孔明'

# Visualization
# Project word vectors into 2D space
rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key):  # index_to_key: index, word
    rawWordVec.append(model.wv[w])  # Word vector
    word2ind[w] = i  # {word: index}
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)

# Plotting
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')

# Plot star chart
# Plot the 2D space projection of all word vectors
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')

# Plot vectors for specific words
words = ['孙权', '刘备', '曹操', '周瑜', '诸葛亮', '司马懿', '汉献帝']
zhfont1 = matplotlib.font_manager.FontProperties(fname='./华文仿宋.ttf', size=16)
# Set Chinese font, otherwise it might display as gibberish
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha=1, color='orange', markersize=10)
        plt.text(xy[0], xy[1], w, fontproperties=zhfont1, alpha=1, color='red')
plt.show()

# Perform some special word vector operations
words = model.wv.most_similar(positive=['玄德', '曹操'], negative=['孔明'])
print("a", words)
words = model.wv.most_similar(positive=['曹操', '蜀'], negative=['魏'])
print("b", words)

