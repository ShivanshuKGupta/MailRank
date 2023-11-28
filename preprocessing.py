# %%

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

# Ensure that 'content' is present in the dataframe
if 'content' in df.columns:
    # Text Cleaning: Remove special characters, numbers, and convert to lowercase
    def clean_text(text):
        # Check for NaN values
        if pd.isnull(text):
            return ""
        
        text = str(text)  # Convert to string if it's not already
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)  # Remove non-alphabetic characters
        text = text.lower()  # Convert to lowercase
        return text

    df['content'] = df['content'].apply(clean_text)

    # Tokenization: Break text into words
    df['content'] = df['content'].apply(word_tokenize)

    # Stopword Removal: Remove common words that don't contribute much to the classification
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(tokens):
        return [word for word in tokens if word.lower() not in stop_words]

        # # Join the lists of tokens back into strings
        # df['content'] = df['content'].apply(lambda x: ' '.join(x))

        # # Apply stop words removal
        # df['content'] = df['content'].apply(remove_stopwords)

    # Save the preprocessed data to a new CSV file
    df.to_csv('preprocessed_data.csv', index=False)

    # Display the first few rows of the preprocessed dataframe
    print("\nPreprocessed Text Data:")
    print(df)
else:
    print("The 'content' column does not exist in the dataframe.")


# %%
