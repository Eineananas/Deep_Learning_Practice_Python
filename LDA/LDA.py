import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# LDA's purpose:
# 1. Output the probability of each topic for each document;
# 2. Output the probability of each word in each topic.

output_path = 'C:/Users/Desktop/lda/result'
file_path = 'C:/Users/Desktop/lda/data'
os.chdir(file_path)
data = pd.read_excel("data.xlsx")  # content type
os.chdir(output_path)
dic_file = "C:/Users/Desktop/lda/stop_dic/dict.txt"
stop_file = "C:/Users/Desktop/lda/stop_dic/stopwords.txt"

def chinese_word_cut(mytext):
    jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file, encoding='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n', 'nz', 'vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)
    word_list = []
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word) < 2:
                find = 1
                break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)
    return (" ").join(word_list)

data["content_cutted"] = data.content.apply(chinese_word_cut)
print(data)

n_features = 1000
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df=0.5,
                                min_df=10)
tf = tf_vectorizer.fit_transform(data.content_cutted)

n_topics = 8
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                doc_topic_prior=0.1,
                                topic_word_prior=0.01,
                                random_state=0)
lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword

n_top_words = 25
tf_feature_names = tf_vectorizer.get_feature_names_out()
topic_word = print_top_words(lda, tf_feature_names, n_top_words)

topics = lda.transform(tf)
topic = []
for t in topics:
    topic.append("Topic #" + str(list(t).index(np.max(t))))
data['概率最大的主题序号'] = topic
data['每个主题对应概率'] = list(topics)
data.to_excel("data_topic1.xlsx", index=False)

# Visualize topics using pyLDAvis (Note: sklearn LDA not supported, use gensim or other implementations)
'''
import pyLDAvis
import pyLDAvis.sklearn

pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')
'''

# Perplexity
plexs = []
n_max_topics = 16
for i in range(1, n_max_topics):
    print(i)
    lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50, random_state=0)
    lda.fit(tf)
    plexs.append(lda.perplexity(tf))

n_t = 15  # Rightmost value for the interval. Note: cannot be greater than n_max_topics
x = list(range(1, n_t + 1))
plt.plot(x, plexs[0:n_t])
plt.xlabel("number of topics")
plt.ylabel("perplexity")
plt.show()
