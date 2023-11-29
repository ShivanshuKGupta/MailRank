# %%

import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

file_path = 'data.csv'
df = pd.read_csv(file_path)

print("Original Data:")
print(df.head())

for col_name in df.columns.drop('target'):
    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = str(text)1
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)  # Remove non-alphabetic characters
        text = text.lower()  # Convert to lowercase
        return text

    df[col_name] = df[col_name].apply(clean_text)
    df[col_name] = df[col_name].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(tokens):
        return [word for word in tokens if word.lower() not in stop_words]

    df[col_name] = df[col_name].apply(remove_stopwords)
    df.to_csv('preprocessed_data.csv', index=False)

    print("\nPreprocessed Text Data:")
    print(df)

# %%
