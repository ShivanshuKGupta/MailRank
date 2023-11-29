# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')
df['content'].fillna('', inplace=True)
df['heading'].fillna('', inplace=True)
df['sender_name'].fillna('', inplace=True)
df['target'].fillna(0, inplace=True)

X = df['content']
y = df['target']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# %%
# Multinomial Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test_tfidf)

# Decode labels if needed
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

accuracies = [accuracy_score(y_test_decoded, y_pred_decoded)]
# Print accuracy and classification report
print("Accuracy:", accuracies[-1])
print("Classification Report:\n", classification_report(
    y_test_decoded, y_pred_decoded))


# %%
# Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
y_pred_rf = rf_classifier.predict(X_test_tfidf)

# Decode labels if needed
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_svm_decoded = label_encoder.inverse_transform(y_pred_svm)
y_pred_rf_decoded = label_encoder.inverse_transform(y_pred_rf)

# Evaluate performance
print("Support Vector Machine:")
accuracies.append(accuracy_score(y_test_decoded, y_pred_svm_decoded))
print("Accuracy:", accuracies[-1])
print("Classification Report:\n", classification_report(
    y_test_decoded, y_pred_svm_decoded))

print("\nRandom Forest:")
accuracies.append(accuracy_score(y_test_decoded, y_pred_rf_decoded))
print("Accuracy:", accuracies[-1])
print("Classification Report:\n", classification_report(
    y_test_decoded, y_pred_rf_decoded))

# %%
plt.plot(['Naive Bayes', 'Support Vector Machine',
         'Random Forest',], accuracies, label='accuracy')
plt.legend()
plt.title('Accuracy vs ML Model')
plt.show()
