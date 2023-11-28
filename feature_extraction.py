#importing libraries for feature extraction
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#reading the data
data = pd.read_csv('preprocessed_text_data.csv')

#creating a bag of words model
cv = CountVectorizer()
X = cv.fit_transform(data['text_column']).toarray()
y = data['label'].values

