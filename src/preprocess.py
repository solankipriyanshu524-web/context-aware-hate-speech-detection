import sys
import types
# Guard against blocked _sqlite3 DLL on Windows systems with Application Control
if 'sqlite3' not in sys.modules:
    sys.modules['sqlite3'] = types.ModuleType('sqlite3')
if '_sqlite3' not in sys.modules:
    sys.modules['_sqlite3'] = types.ModuleType('_sqlite3')

import re
import string
from nltk.stem.porter import PorterStemmer
try:
    from nltk.corpus import stopwords
except Exception:
    stopwords = None

# Safe fallback stopwords
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

stemmer = PorterStemmer()
try:
    stopword_set = set(stopwords.words('english'))
except Exception:
    stopword_set = FALLBACK_STOPWORDS

def deobfuscate(text):
    text = str(text).lower()
    text = text.replace('@', 'a').replace('4', 'a')
    text = text.replace('$', 's').replace('5', 's')
    text = text.replace('0', 'o')
    text = text.replace('1', 'i').replace('!', 'i')
    text = text.replace('3', 'e')
    return text

def collapse_dups(text):
    # Condense consecutive characters: fuuuck -> fuck, madarchood -> madarchod
    return re.sub(r'(.)\1+', r'\1', text)

def clean_text(text):
    """
    Cleans the input text for Hate Speech Detection.
    - Converts to lowercase and normalizes Leetspeak (special characters)
    - Collapses elongated words (e.g. coooool -> col)
    - Removes URLs, HTML tags
    - Removes punctuation and special characters
    - Removes stopwords
    - Applies Stemming
    """
    text = deobfuscate(text)
    text = collapse_dups(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Remove stopwords and stem
    words = [stemmer.stem(word) for word in text.split(' ') if word not in stopword_set and word != '']
    return " ".join(words)
