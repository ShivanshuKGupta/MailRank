import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources (punkt tokenizer and stopwords)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load your CSV file
file_path = 'data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand the structure of the data
print("Original Data:")
print(df.head())

# Text Cleaning: Remove unnecessary characters, HTML tags, and special symbols
def clean_text(text):
    # Check for NaN values
    if pd.isnull(text):
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text

# Apply text cleaning to the desired column (replace 'content' with your actual column name)
df['content'] = df['content'].apply(clean_text)

# Tokenization: Break text into words
df['content'] = df['content'].apply(word_tokenize)

# Stopword Removal: Remove common words that don't contribute much to the classification
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

df['content'] = df['content'].apply(remove_stopwords)

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_data.csv', index=False)

# Display the first few rows of the preprocessed dataframe
print("\nPreprocessed Data:")
print(df.head())
