# %%
from nltk.stem import PorterStemmer
import pandas as pd

df = pd.read_csv('preprocessed_text_data.csv')


def apply_stemming(txt):
    words = txt.split()


stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]

stemmed_words
