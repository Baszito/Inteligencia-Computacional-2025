from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

vocabulary = {}
n = 1
data = pd.read_csv("TRABAJO CREATIVO\SherLockFakenewsProcessedNoStopWords.csv")
data[['news_headline']] = data[['news_headline']].fillna('')

for i in range(0, data.shape[0]):
    txt = data.iloc[i, 0]
    tokens = word_tokenize(txt)
    for t in tokens:
        if t in vocabulary:
            continue
        vocabulary[t] = n
        n+=1
df_vocabulary = pd.DataFrame(list(vocabulary.items()), columns=["word", "index"])
df_vocabulary.to_csv(r'TRABAJO CREATIVO\vocabulary_nostopwords.csv', 
          index=False,
          encoding='utf-8',     # Codificación
          header=False)          # Incluir nombres de columnas