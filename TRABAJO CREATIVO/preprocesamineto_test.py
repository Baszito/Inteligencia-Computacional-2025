import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#nltk.download('stopwords')
#nltk.download('punkt')


#data = pd.read_csv("TRABAJO CREATIVO\SherLockFakenewsNetOriginal.csv")
data = pd.DataFrame([['<h1>hi world</h1>', 1.0],
                    [np.nan, np.nan],
                    ['hello what is happening right now [VIDEO]', 0.0],
                    ['<p>  Around 200 arrests made over election violence [NO CREERAS LO QUE VERAS A CONTINUACION] sambar [IMAGE] </p>', 0.0],
                    ['You can see more in https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element', 1.0],
                    ['ItÃ¢â‚¬â„¢s About Time! Twitter Just Kicked Off A Bunch Of Alt-Right Accounts For Hate Speech', 1.0]], columns=list('AB'))

# Hay un paso que aparentemente es agarrar el 50% del dataset para acelerar el proceso.

# Dejo la línea comentada para posiblemente trabajarlo en un futuro, pero por simplicidad voy a trabajar con todos los datos
#data = data.sample(n=108341, random_state=1)

print(data)

#print(data['reliable'])

# Se reemplazan los datos vacíos o nulos con fillna()
data[['B']] = data[['B']].fillna(value=0)
data[['A']] = data[['A']].fillna('')

#print(data['reliable'])
def fix_encoding(text: str):
    try:
        return text.encode('latin1').decode('utf-8')
    except UnicodeEncodeError:
        return text
# Si después de la decodificación sigue medio garchado el texto, esto devuelve True
def has_weird_chars(text):
    # permite letras, números, espacios, signos comunes y caracteres acentuados válidos
    pattern = r"^[a-zA-Z0-9\s\.,;:'\"!?¡¿\-\(\)áéíóúÁÉÍÓÚñÑüÜ]*$"
    return not bool(re.match(pattern, text))
# EL siguiente for es para eliminar los tags HTML
for i in range(0, data.shape[0]):
    #print("############\n")
    txt = data.iloc[i, 0]
    txt = fix_encoding(txt)



    soup = BeautifulSoup(txt, features="html.parser")
    txt = soup.get_text()

    # La siguiente línea es para eliminar elementos entre corchetes, incluidos los corchetes
    txt = re.sub(r'\[.*?\]', '', txt)

    txt = re.sub(r'http[s]?://\S+', '', txt)

    if (has_weird_chars(txt)):
        txt = ""

    data.iloc[i, 0] = txt

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(txt.lower())
   
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    print(filtered_tokens)
    txt = ""
    for f in filtered_tokens:
        txt = txt + f + " "

    data.iloc[i, 0] = txt
    #print(data.iloc[i, 0])
    #for j in range(0, data.shape[1]):
    #    print(data.iloc[i, j])

print(data)