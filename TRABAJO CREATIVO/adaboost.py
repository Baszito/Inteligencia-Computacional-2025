from sklearn.ensemble import AdaBoostClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_withstopwords = pd.read_csv('TRABAJO CREATIVO/SherLockFakenewsProcessedWithStopWords.csv')
embedding_matrix = pd.read_csv('TRABAJO CREATIVO/SherLockFakenewsProcessedWithStopWords.csv')

X = data_withstopwords[['news_headline']]
y = data_withstopwords[['reliable']]

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, shuffle=True)

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)

# ToDo: Pasar los datos a los embeddings

model1 = abc.fit(X_trn, y_trn)

y_pred = model1.predict(X_tst)

print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_tst, y_pred))

