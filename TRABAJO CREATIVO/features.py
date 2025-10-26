import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#Dataset Spiltting
data_stopwords = pd.read_csv('TRABAJO CREATIVO/SherLockFakenewsProcessedWithStopWords.csv')
X = data_stopwords[['news_headline']]
y = data_stopwords[['reliable']]
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)
#print(X_trn)


#Tokenization and Padding
max_len = max([len(str(text).split()) for text in X['news_headline']])
tokenizer = Tokenizer(num_words=max_len,oov_token='<OOV>')
tokenizer.fit_on_texts(X['news_headline'].tolist())

X_trn_sequences=tokenizer.texts_to_sequences(X_trn['news_headline'].tolist())
X_trn_padded=pad_sequences(X_trn_sequences,padding='post')
#print(X_trn_padded)
X_tst_sequences=tokenizer.texts_to_sequences(X_tst['news_headline'].tolist())
X_tst_padded=pad_sequences(X_tst_sequences,padding='post')

print(X_trn_padded)
print("##########################")
print(X_tst_padded)

y_trn_arr = y_trn.values
y_tst_arr = y_tst.values
#Word Embeddings

#Embedding Matrix Initialization

#Integration with Tokenizer
