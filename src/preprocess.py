import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download only if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords + apply stemming
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    
    return " ".join(words)