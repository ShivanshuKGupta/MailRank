# %%
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# %%
df = pd.read_csv('data.csv')

print("Original Data:")
print(df.head())


def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    text = text.lower()
    return text


for col_name in df.columns.drop(['target', 'sender_email']):
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
