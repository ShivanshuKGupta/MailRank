# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from analysis import generate_report, save_model

df = pd.read_csv('preprocessed_data.csv')
df['content'].fillna('', inplace=True)
df['heading'].fillna('', inplace=True)
df['sender_name'].fillna('', inplace=True)
df['target'].fillna(0, inplace=True)

X = df['content']+' '+df['heading']
y = df['target']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# %%
# Multinomial Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

y_pred = nb_classifier.predict(X_test_tfidf)

y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

accuracies = [accuracy_score(y_test_decoded, y_pred_decoded)]

generate_report(y_test_decoded, y_pred_decoded, "Naive Bayes")

# %%
# Support Vector Machine (SVM)
svm_classifier = SVC(kernel='poly')
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
y_pred_rf = rf_classifier.predict(X_test_tfidf)

y_pred_svm_decoded = label_encoder.inverse_transform(y_pred_svm)
y_pred_rf_decoded = label_encoder.inverse_transform(y_pred_rf)
generate_report(y_test_decoded, y_pred_svm_decoded, "SVM")
generate_report(y_test_decoded, y_pred_rf_decoded, "Random Forest")

print("Support Vector Machine:")
accuracies.append(accuracy_score(y_test_decoded, y_pred_svm_decoded))

print("\nRandom Forest:")
accuracies.append(accuracy_score(y_test_decoded, y_pred_rf_decoded))

# %%
plt.plot(['Naive Bayes', 'Support Vector Machine',
         'Random Forest',], accuracies, label='accuracy')
plt.legend()
plt.title('Accuracy vs ML Model')
plt.show()
