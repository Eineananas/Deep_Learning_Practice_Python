## Chinese Text Sentiment Analysis — Based on Bag-of-Words Model and Naive Bayes Classifier

import numpy as np
import pandas as pd

# Read data from Excel file
data = pd.read_excel('data_test_train.xlsx')
print(data.head())

## Data Preprocessing
# Remove duplicates and stopwords (not done yet)

## Tokenization using jieba
import jieba

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

# Tokenize Chinese text into words and join these words with spaces, returning a string of segmented words
data['cut_comment'] = data.comment.apply(chinese_word_cut)
print(data)

## Feature Extraction using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = '哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = CountVectorizer(max_df=0.8,
                       min_df=3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=stopwords
                       )

## Data Splitting
X = data['cut_comment']
y = data.sentiment

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

## Feature Representation
show = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names_out())
print("show head:", show.head())
print("show shape:", show.shape)

## Naive Bayes Classification
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
X_train_vect = vect.fit_transform(X_train)
print("X_train_vect shape:", X_train_vect.shape)
print("y_train shape:", y_train.shape)
nb.fit(X_train_vect, y_train)

# Training accuracy
train_score = nb.score(X_train_vect, y_train)
print("Training Accuracy:", train_score)

# Test accuracy
X_test_vect = vect.transform(X_test)
test_score = nb.score(X_test_vect, y_test)
print("Test Accuracy:", test_score)

## Prediction on New Data
data = pd.read_excel("unlabelled_data.xlsx")
data['cut_comment'] = data.comment.apply(chinese_word_cut)
X = data['cut_comment']
X_vec = vect.transform(X)
nb_result = nb.predict(X_vec)

# Add predicted sentiment to the dataframe
data['nb_result'] = nb_result
print(data)
