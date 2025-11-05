import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

#Divison del dataset
def make_embedding_matrix(glove_path, processed_data_path, embedding_matrix_path):
    print("Creando matriz de embeddings...")
    data = pd.read_csv(processed_data_path)
    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            #print(word)
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

<<<<<<< Updated upstream
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
=======
    embedding_dim = 100
>>>>>>> Stashed changes

    #Dividimos el dataset
    X = data[['news_headline']]


    #Tokenizacion y Padding
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
            # Palabra no encontrada -> vector aleatorio
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

    np.savetxt(embedding_matrix_path, embedding_matrix, delimiter=',')