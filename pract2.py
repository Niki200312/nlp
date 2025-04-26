import nltk
nltk.download('punkt_tab')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Sample data
data = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Bag-of-Words Approach: Count Occurrence
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(data)
count_feature_names = count_vectorizer.get_feature_names_out()
count_array = count_matrix.toarray()

print("Count Occurrence:")
print(count_feature_names)
print(count_array)

# Bag-of-Words Approach: Normalized Count Occurrence
normalized_count_array = count_array / count_array.sum(axis=1, keepdims=True)
print("\nNormalized Count Occurrence:")
print(normalized_count_array)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()

print("\nTF-IDF:")
print(tfidf_feature_names)
print(tfidf_array)

# Word Embeddings using Word2Vec
tokenized_data = [word_tokenize(sentence.lower()) for sentence in data]
word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = [word2vec_model.wv[word] for sentence in tokenized_data for word in sentence]

print("\nWord Embeddings using Word2Vec:")
print(word_vectors)
