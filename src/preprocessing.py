"""
:author: Nikola Pizurica

This module contains a class whose objects can be passed to CountVectorizer and TfidfVectorizer
constructors for custom tokenizing (along with preprocessing), or used in custom vectorizers.
"""

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re


class LemmaTokenizer(object):
    """
    Performs noun lemmatizing by default and can optionally lemmatize other parts of speech, but
    that increases the runnning time drastically. Also, punctuation symbols and numbers are
    removed by default, but they can be kept.
    """
    def __init__(self, keep_numbers=False, keep_punctuation=False, use_pos_tagging=False):
        """
        :param keep_numbers:        If True, numbers that appear in texts will be treated as valid tokens.

        :param keep_punctuation:    If True, punctuation symbols will be treated as valid tokens.

        :param use_pos_tagging:     If True, parts of speech other than nouns will also be lemmatized.
        """
        self.wnl = WordNetLemmatizer()
        self.keep_numbers = keep_numbers
        self.keep_punctuation = keep_punctuation
        self.use_pos_tagging = use_pos_tagging
        self.tags = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

    def clean(self, text):
        """
        :param text:    A string to be processed.

        :return:        Lowercase cleaned form of text.
        """
        if not self.keep_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        if not self.keep_numbers:
            text = re.sub(r'\b\d+\b', ' ', text)
        return text.lower()

    def pos(self, word):
        """
        :param word:    A word/token/term.

        :return:        Wordnet part of speech tag for word.
        """
        tag = pos_tag([word])[0][1][0].upper()
        return self.tags.get(tag, wordnet.NOUN)

    def lemma_tokenize(self, text):
        """
        :param text:    A string to be tokenized.

        :return:        A list of lemmatized tokens.
        """
        if self.use_pos_tagging:
            return [self.wnl.lemmatize(t, self.pos(t)) for t in word_tokenize(self.clean(text))]
        else:
            return [self.wnl.lemmatize(t) for t in word_tokenize(self.clean(text))]

    def normalize_stopwords(self, stopwords):
        """
        :param stopwords:   A list of stopwords being used.

        :return:            Normalized stopword list, i.e. the one where each stoword has been lemmatized.
        """
        temp_words = []
        for w in stopwords:
            temp_words += word_tokenize(self.clean(w))
        if self.use_pos_tagging:
            return [self.wnl.lemmatize(w, self.pos(w)) for w in temp_words]
        else:
            return [self.wnl.lemmatize(w) for w in temp_words]

    def __call__(self, text):
        """
        Necessary for vectorizer compatibility (so we can pass an object of this class to TfidfVectorizer).
        :param text:    A string to be processed.

        :return:        A list of tokens.
        """
        if self.use_pos_tagging:
            return [self.wnl.lemmatize(t, self.pos(t)) for t in word_tokenize(self.clean(text))]
        else:
            return [self.wnl.lemmatize(t) for t in word_tokenize(self.clean(text))]
