import pandas as pd
import nltk
import inflect
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('data/reviews.csv')

def to_lowercase(words):
    # Convert all characters to lowercase from list of tokenized words
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    # Remove punctuation from list of tokenized words
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_numbers(words):
    # Replace all interger occurrences in list of tokenized words with textual representation
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = re.sub('[0-9]+', '', word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    # Remove stop words from list of tokenized words
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):
    # Lemmatize verbs in list of tokenized words
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    # Data Cleansing Pipeline
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_numbers(words)
    words = remove_stopwords(words)
    lemmas = lemmatize_verbs(words)
    return lemmas



word_list = []
word_list_dict = {}
for idx, row  in df.iterrows():
    words = nltk.word_tokenize(row['content'])
    words = normalize(words)
    word_list.append(words)
    word_list_dict[idx] = word_list
