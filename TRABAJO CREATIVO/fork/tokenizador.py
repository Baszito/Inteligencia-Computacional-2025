from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

vocabulary = pd.read_csv(r'TRABAJO CREATIVO\vocabulary_nostopwords.csv')

word_to_index = {row[0]: idx+2 for idx, row in vocabulary.iterrows()}  # +1 por el padding 0

def text_to_sequence(text):
    tokens = word_tokenize(text.lower())
    return [word_to_index.get(word, 0) for word in tokens]  # 0 si no está

print(text_to_sequence('Hello world'))