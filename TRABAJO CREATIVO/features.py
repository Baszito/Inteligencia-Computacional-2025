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
matriz_embedding = pd.read_csv('TRABAJO CREATIVO/embedding_matrixK.csv', header=None)
matriz_embedding = matriz_embedding.to_numpy()

#Tokenization and Padding
max_len = max([len(str(text).split()) for text in X['news_headline']])
tokenizer = Tokenizer(num_words=max_len,oov_token='<OOV>')
tokenizer.fit_on_texts(X['news_headline'].tolist())
X_trn_sequences=tokenizer.texts_to_sequences(X_trn['news_headline'].tolist())
X_trn_padded=pad_sequences(X_trn_sequences,padding='post')

X_tst_sequences=tokenizer.texts_to_sequences(X_tst['news_headline'].tolist())
X_tst_padded=pad_sequences(X_tst_sequences,padding='post')

#X_trn_padded_v = X_trn_padded.flatten()
embedding_dim = 100
seq_len = len(X_trn_padded[0])


mat = np.zeros((len(X_trn_padded), seq_len * embedding_dim))
i = 0
for x in X_trn_padded:
   #v = np.zeros((0, 0))
    vecs = []
    for token in x:
        vecs.append(matriz_embedding[token])
        #v_e = matriz_embedding[token]
        #v = np.concatenate(v, v_e)
    v = np.concatenate(vecs)
    mat[i] = v
    i+=1
np.savetxt(r'TRABAJO CREATIVO\TRAINING_DATA_NUMBER.csv', mat, delimiter=',')
    
#print(X_trn_padded)
#print("##########################")
#print(X_tst_padded)

y_trn_arr = y_trn.values
y_tst_arr = y_tst.values
#Word Embeddings

#Embedding Matrix Initialization

#Integration with Tokenizer


