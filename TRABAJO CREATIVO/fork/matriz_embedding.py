from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

vocabulary = pd.read_csv(r'TRABAJO CREATIVO\vocabulary_nostopwords.csv')
#Dataset Spiltting

data_nostopwords = pd.read_csv('TRABAJO CREATIVO\SherLockFakenewsProcessedNoStopWords.csv')
embeddings_index = {}
with open(r"C:\Users\valentin\Desktop\fuckultad\Inteligencia Computacional\glove.twitter.27B\glove.twitter.27B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        #print(word)
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector

X = data_nostopwords[['news_headline']]
y = data_nostopwords[['reliable']]
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

vocab_size = len(vocabulary) + 1  # +1 por el padding
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))


for i in range(1, vocabulary.shape[0]+1):
    word = vocabulary.iloc[i-1, 0]
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        # Palabra no encontrada → vector aleatorio
        embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

np.savetxt(r'TRABAJO CREATIVO\embedding_matrix.csv', embedding_matrix, delimiter=',')



#word_to_index = {row[0]: idx+1 for idx, row in vocabulary.iterrows()}  # +1 por el padding 0




#Word Embeddings

#Embedding Matrix Initialization

#Integration with Tokenizer