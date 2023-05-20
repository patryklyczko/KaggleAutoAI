from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, wordnet
from collections import Counter
from spellchecker import SpellChecker
import nltk
import pandas as pd
import emoji
import string
import re


class TextPreprocessing():
    def __init__(self, text:pd.DataFrame):
        self.text = text

    def lowercase(self):
        self.text = self.text.apply(lambda x: lower(x))

    def punctuation(self):
        self.text = self.text.apply(lambda x: remove_punctuation(x))

    def stopwords(self, language='english'):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        stopword = set(stopwords.words(language))
        self.text = self.text.apply(lambda x: remove_stopwords(x, stopword))

    def frequent_words(self, most_common=10):
        words = Counter()
        for text in self.text:
            for word in text.split():
                words[word] += 1
        return words.most_common(most_common)
    
    def frequent_words(self, number = 0, words=None, auto=False):
        if auto:
            frequent_words = self.frequent_words(number)
            frequent_words = set(word for  (word, _) in frequent_words)
        else:
            frequent_words = words
        self.text = self.text.apply(lambda x: remove_frequent_words(x, frequent_words))

    def special_characters(self, change_into=" ",syntax='[^a-zA-Z0-9]'):
        self.text = self.text.apply(lambda x: remove_special_characters(x, change_into, syntax))

    def stemming(self):
        ps = PorterStemmer()
        self.text = self.text.apply(lambda x: stemming(x, ps))

    def lemmatization(self):
        lemmat = WordNetLemmatizer()
        wordnet_map = {
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "J": wordnet.ADJ,
            "R": wordnet.ADV,
        }
        self.text = self.text.apply(lambda x: lemat_word(x, lemmat, wordnet_map))

    def url(self, change_to=""):
        self.text = self.text.apply(lambda x: remove_url(x, change_to))

    def html(self):
        self.text = self.text.apply(lambda x: remove_html(x))

    def spelling(self):
        spell = SpellChecker()
        self.text = self.text.apply(lambda x: correct_spelling(x, spell))

    def emoji(self, lang='en'):
        self.text = self.text.apply(lambda x: strip_emoji(x, lang))

    def get(self):
        return self.text


def lower(text):
    return text.lower()

def strip_emoji(text, lang):
    return emoji.demojize(text, language=lang)

def remove_punctuation(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans(' ', ' ', punctuations))
    
def remove_stopwords(text, stopwords):
    return " ".join([word for word in text.split() if word not in stopwords])

def remove_frequent_words(text, remove_words):
    return " ".join([word for word in text.split() if word not in remove_words])

def remove_special_characters(text, change_into, syntax):
    text = re.sub(syntax,change_into, text)
    text = re.sub('\s+', ' ', text)
    return text

def stemming(text, stemmer):
    return " ".join([stemmer.stem(word) for word in text.split()])

def lemat_word(text, lemat, wordnet_map):
    pos_text = pos_tag(text.split())
    return " ".join([lemat.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])

def remove_url(text, change_to):
    return re.sub(r'https?://\S+|www\.\S+', change_to, text)

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def correct_spelling(text, spell):
    corrected_text = []
    mispelled = spell.unknown(text.split())
    for word in text.split():
        if word in mispelled:
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_text.append(corrected_word)
            else:
                corrected_text.append(word)
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
