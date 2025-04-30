import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sample data
data = {
    'text': ["This is the first document.",
             "This document is the second document.",
             "And this is the third one.",
             "Is this the first document?"],
    'label': ['A', 'B', 'C', 'A']
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Text Cleaning, Lemmatization, and Stop Words Removal
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [token for token in tokens if token.isalnum()]  # Remove non-alphanumeric characters
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Label Encoding
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# TF-IDF Representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Save Outputs
df[['clean_text', 'label', 'encoded_label']].to_csv('cleaned_data.csv', index=False)
tfidf_df.to_csv('tfidf_representation.csv', index=False)

print("Outputs saved successfully.")
