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


def preprocess_text(text):
    # tokens = word_tokenize(text)
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word.lower() not in stop_words]

    print(f"{text=}")
    tokens = text
    print(f"{tokens=}")
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text


columns_to_preprocess = df.columns.drop('target')

for column in columns_to_preprocess:
    df[column] = df[column].apply(preprocess_text)

# df
