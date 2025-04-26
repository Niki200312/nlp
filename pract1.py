import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer, TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Sample text
text = "Tokenization is a key task in NLP. It breaks text into tokens, which can be words, phrases, or symbols."

# Tokenization
# Whitespace Tokenizer
whitespace_tokenizer = WhitespaceTokenizer()
whitespace_tokens = whitespace_tokenizer.tokenize(text)
print("Whitespace Tokenizer:", whitespace_tokens)

# Punctuation-based Tokenizer
punct_tokenizer = WordPunctTokenizer()
punct_tokens = punct_tokenizer.tokenize(text)
print("Punctuation-based Tokenizer:", punct_tokens)

# Treebank Tokenizer
treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(text)
print("Treebank Tokenizer:", treebank_tokens)

# Tweet Tokenizer
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(text)
print("Tweet Tokenizer:", tweet_tokens)

# Multi-Word Expression Tokenizer
mwe_tokenizer = MWETokenizer()
mwe_tokenizer.add_mwe(("key", "task"))
mwe_tokens = mwe_tokenizer.tokenize(text.split())
print("MWE Tokenizer:", mwe_tokens)

# Stemming
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")

porter_stems = [porter_stemmer.stem(token) for token in treebank_tokens]
print("Porter Stemmer:", porter_stems)

snowball_stems = [snowball_stemmer.stem(token) for token in treebank_tokens]
print("Snowball Stemmer:", snowball_stems)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in treebank_tokens]
print("Lemmatization:", lemmatized_tokens)
