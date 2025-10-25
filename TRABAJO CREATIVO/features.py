import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#Dataset Spiltting
data_stopwords = pd.read_csv('TRABAJO CREATIVO/SherLockFakenewsProcessedWithStopWords.csv')
X = data_stopwords[['news_headline']].astype('str')
y = data_stopwords[['reliable']]

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

#print(X_tst)

#Tokenization and Padding
tokenizer = Tokenizer(num_words=1000,oov_token="<OOV>")
tokenizer.fit_on_texts(X_trn)
sequences=tokenizer.texts_to_sequences(X_trn)
padded=pad_sequences(sequences,padding="post")
print(sequences)

#Word Embeddings

#Embedding Matrix Initialization

#Integration with Tokenizer
