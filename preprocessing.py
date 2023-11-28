# %%
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (stopwords and punkt tokenizer)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load your CSV file
file_path = 'data.csv'
df = pd.read_csv(file_path)

# Display the column names in the dataframe
print("Column Names:")
print(df.columns)

# Display the first few rows of the dataframe to understand the structure of the data
print("\nOriginal Data:")
print(df.head())

# Ensure that 'target' is present in the dataframe
if 'target' in df.columns:
    # Text Cleaning: Remove special characters, numbers, and convert to lowercase
    def clean_text(text):
        # Check for NaN values
        if pd.isnull(text):
            return ""
        
        text = str(text)  # Convert to string if it's not already
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)  # Remove non-alphabetic characters
        text = text.lower()  # Convert to lowercase
        return text

    df['target'] = df['target'].apply(clean_text)

    # Tokenization: Break text into words
    df['target'] = df['target'].apply(word_tokenize)

    # Handling Stop Words: Remove common words that do not contribute much to the meaning
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(tokens):
        return [word for word in tokens if word.lower() not in stop_words]

    # Join the lists of tokens back into strings
    df['target'] = df['target'].apply(lambda x: ' '.join(x))

    # Apply stop words removal
    df['target'] = df['target'].apply(remove_stopwords)

    # Save the preprocessed data to a new CSV file
    df.to_csv('preprocessed_text_data.csv', index=False)

    # Display the first few rows of the preprocessed dataframe
    print("\nPreprocessed Text Data:")
    print(df)
else:
    print("The 'target' column does not exist in the dataframe.")


# %%
