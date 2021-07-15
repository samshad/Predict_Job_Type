"""import spacy
nlp = spacy.load("en_core_web_trf")
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')

from dateutil import parser
import string
import re
from nltk.stem.porter import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS

from gensim.models import KeyedVectors"""

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import text_cleaner as h
import langdetect
from googletrans import Translator
translator = Translator()


train = pd.read_csv('Data/test_jobs.csv', encoding="utf-8")
# train['lang'] = train["clean_description"].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")

cnt = 1
for index, row in train.iterrows():
    print(f"{cnt} -> {row['lang']}")
    text = str(row['clean_description'])
    arr = []
    if row['lang'] != 'en':
        print(text)
        es = translator.translate(text, dest='en').text
        print(es)
        arr.append([row['title'], row['judi'], row['category'], row['lang'], es])
    else:
        arr.append([row['title'], row['judi'], row['category'], row['lang'], row['clean_description']])
    if cnt == 1:
        df = pd.DataFrame(arr, columns=['title', 'judi', 'category', 'lang', 'clean_description'])
        df.to_csv('Data/translated_file.csv', index=False)
    else:
        df = pd.DataFrame(arr, columns=['title', 'judi', 'category', 'lang', 'clean_description'])
        df.to_csv('Data/translated_file.csv', index=False, header=False, mode='a')
    cnt += 1

