# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from preprocessing import preprocess

preprocess(file_name='new_data.csv',
           preprocessed_file='new_data_preprocessed.csv')
df = pd.read_csv('new_data_preprocessed.csv')

X = df['content']
# %%
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_np = X_tfidf.toarray()
# %%
X_gru = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])

loaded_model = load_model('gru_model.h5')

y_pred = loaded_model.predict(X)
print(f"{y_pred=}")
