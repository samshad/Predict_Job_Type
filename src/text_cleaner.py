import spacy
nlp = spacy.load("en_core_web_trf")
# python -m spacy download en_core_web_trf
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

from dateutil import parser
import string
import re
from nltk.stem.porter import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS

from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings('ignore')


def is_valid_date(date_str):
    try:
        parser.parse(date_str)
        return True
    except:
        return False


def date_removal(data):
    new_list = [' '.join([w for w in line.split() if not is_valid_date(w)]) for line in data]
    return new_list[0]


def stemmer_and_stopWord(doc):
    doc = nlp(doc)
    token_list = []
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text
        token_list.append(lemma)

    stemmed = token_list

    # Create list of word tokens after removing stopwords

    filtered_sentence = []
    for word in stemmed:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)
    return ' '.join(filtered_sentence)


def normaliz(filtered_sentence):
    words = [str(word).lower() for word in filtered_sentence.split()]
    return ' '.join(words)


def numbers_removal(data):
    s = [data]
    result = ''.join([i for i in s if not i.isdigit()])
    return result


def punch_removal(words):
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in [words]]
    return re.sub(' +', ' ', stripped[:100][0])


def cleaner(data):
    string = [data]
    string = date_removal(string)
    # string = numbers_removal(string)
    string = punch_removal(string)
    string = stemmer_and_stopWord(string)
    string = normaliz(string)
    return string
