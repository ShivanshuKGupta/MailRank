#importing libraries for feature extraction
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#reading the data
data = pd.read_csv('preprocessed_data.csv')

#creating a bag of words model
cv = CountVectorizer()
X = cv.fit_transform(data['text_column']).toarray()
y = data['label'].values

#3.1.2. TF-IDF
#creating a TF-IDF model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['text_column']).toarray()
y = data['label'].values

#3.1.3. Word Embeddings
#loading the pre-trained word embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]  # The first value is the word, the rest are the values of the embedding
        coefs = np.asarray(values[1:], dtype='float32')  # Load embedding
        embeddings_index[word] = coefs  # Add embedding to our embedding dictionary
print('Found %s word vectors.' % len(embeddings_index))

#creating a word embeddings matrix
embedding_matrix = np.zeros((len(embeddings_index), 100))
for i, word in enumerate(embeddings_index):
    embedding_vector = embeddings_index[word]
    embedding_matrix[i] = embedding_vector
    
#creating a word embeddings model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(len(embeddings_index), 100, weights=[embedding_matrix], input_length=1000, trainable=False))
model.compile('rmsprop', 'mse')



