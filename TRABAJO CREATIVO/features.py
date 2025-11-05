import os,warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message=".*oneDNN custom operations.*")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#Dataset Spiltting

noStopWordsBAND = True

class Features:
    def __init__(self):
        if noStopWordsBAND:
            data_stopwords = pd.read_csv('TRABAJO CREATIVO/SherLockFakenewsProcessedNoStopWords.csv')
            matriz_embedding = pd.read_csv('TRABAJO CREATIVO/embedding_matrixKNoStopwords.csv', header=None)
        else:
            data_stopwords = pd.read_csv('TRABAJO CREATIVO/SherLockFakenewsProcessedWithStopWords.csv')
            matriz_embedding = pd.read_csv('TRABAJO CREATIVO/embedding_matrixKWithStopwords.csv', header=None)
        X = data_stopwords[['news_headline']]
        y = data_stopwords[['reliable']]
        X_trn, X_tst, self.y_trn, self.y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)
        #print(X_trn)
        
        self.matriz_embedding = matriz_embedding.to_numpy()

        #Tokenization and Padding
        self.max_len = max([len(str(text).split()) for text in X['news_headline']])
        tokenizer = Tokenizer(num_words=self.max_len,oov_token='<OOV>')
        tokenizer.fit_on_texts(X['news_headline'].tolist())
        X_trn_sequences=tokenizer.texts_to_sequences(X_trn['news_headline'].tolist())
        self.X_trn_padded = pad_sequences(X_trn_sequences, maxlen=self.max_len, padding='post', truncating='post')
        X_tst_sequences=tokenizer.texts_to_sequences(X_tst['news_headline'].tolist())
        self.X_tst_padded = pad_sequences(X_tst_sequences, maxlen=self.max_len, padding='post', truncating='post')

        self.y_trn = self.y_trn.to_numpy()
        self.y_tst = self.y_tst.to_numpy()



