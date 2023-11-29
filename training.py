# %%
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
df = pd.read_csv('preprocessed_data.csv')


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(tokens):
    tokens = tokens.replace('[', '')
    tokens = tokens.replace(']', '')
    tokens = tokens.replace('\'', '')
    tokens = tokens.replace(',', '')
    tokens = tokens.split(' ')
    # print(f"{tokens=}")

    stemmed_words = [stemmer.stem(word) for word in tokens]
    # print(f"{stemmed_words=}")

    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
    # print(f"{lemmatized_words=}")

    preprocessed_text = ' '.join(lemmatized_words)
    # print(f"{preprocessed_text=}")

    return preprocessed_text


columns_to_preprocess = df.columns.drop(['target', 'sender_email'])

for column in columns_to_preprocess:
    df[column] = df[column].apply(preprocess_text)


def remove_braces(txt):
    print(f"{txt=}")
    if (not isinstance(txt, str) or txt.count('<') == 0):
        return txt
    txt = txt.replace('<', '')
    txt = txt.replace('>', '')
    return txt


df['sender_email'] = df['sender_email'].apply(remove_braces)
df.to_csv('prerocessed_data.csv')
