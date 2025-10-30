from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#vocabulary = pd.read_csv(r'TRABAJO CREATIVO\vocabulary_nostopwords.csv')
#Dataset Spiltting

data_nostopwords = pd.read_csv('TRABAJO CREATIVO\SherLockFakenewsProcessedNoStopWords.csv')
embeddings_index = {}
with open(r"C:\Users\lucga\OneDrive\Documentos\GitHub\glove.twitter.27B\glove.twitter.27B.100d.txt", encoding="utf-8") as f:
#with open(r"D:\Cosas de la cufa\Inteligencia Computacional\glove.twitter.27B.100d.txt", encoding="utf-8") as f:
#with open(r"C:\Users\valentin\Desktop\fuckultad\Inteligencia Computacional\glove.twitter.27B\glove.twitter.27B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        #print(word)
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector

#X = data_nostopwords[['news_headline']]
#y = data_nostopwords[['reliable']]
#X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

#vocab_size = len(vocabulary) + 1  # +1 por el padding
embedding_dim = 100


#Dataset Spiltting
#data_stopwords = pd.read_csv('TRABAJO CREATIVO/SherLockFakenewsProcessedWithStopWords.csv')
X = data_nostopwords[['news_headline']]
y = data_nostopwords[['reliable']]
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)
#print(X_trn)


#Tokenization and Padding
max_len = max([len(str(text).split()) for text in X['news_headline']])
tokenizer = Tokenizer(num_words=max_len,oov_token='<OOV>')
tokenizer.fit_on_texts(X['news_headline'].tolist())

vocab_size = len(tokenizer.word_index) + 1  # +1 por el padding
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        # Palabra no encontrada → vector aleatorio
        embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

np.savetxt(r'TRABAJO CREATIVO\embedding_matrixKNoStopwords.csv', embedding_matrix, delimiter=',')