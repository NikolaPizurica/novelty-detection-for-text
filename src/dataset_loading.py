"""
:author: Nikola Pizurica

Loading datasets containing text documents and converting those documents into a desired numeric
representation that can be fed into machine learning models.
"""

from src.preprocessing import LemmaTokenizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.datasets import load_files
from nltk.corpus import reuters
from numpy import array
from random import random


class ReutersLoader:
    """
    Used to load Reuters benchmark dataset using nltk module. Multilabeled text documents are filtered out.
    """
    def __init__(self, vectorizer, binarize_labels=False):
        """
        :param vectorizer:      Used to convert raw text documents (strings) into vectors of numbers. It
                                can be one of vectorizers available in sklearn, or a custom vectorizer, as
                                long as it provides fit_transform and transform functions in the same
                                format as they are defined in sklearn.

        :param binarize_labels: Whether to use default label encoding (using numbers 0,...,n-1 to represent
                                n classes of documents) or to convert them to binary representation (using
                                [1 0 ... 0 0] for the first class, ..., [0 0 ... 0 1] for the last one).
        """
        # A list of training class names (categories to which a document may belong).
        self.train_classes = []
        # A list of test class names (categories to which a document may belong).
        self.test_classes = []
        # A list of filenames (id's) in the training set.
        self.train = []
        # A list of filenames (id's) in the test set.
        self.test = []
        # A dictionary that contains the entire dataset, converted into a numeric representation.
        self.data = {'x_train': [], 'y_train': [], 'x_test': [], 'y_test': []}
        self.vectorizer = vectorizer
        if binarize_labels:
            self.labeler = LabelBinarizer()
        else:
            self.labeler = LabelEncoder()

    def load_data(self, train_classes=[], test_classes=[], other_frac=None):
        """
        :param train_classes:   If left empty, the function loads training documents from all classes
                                (categories). Otherwise, it loads only those classes that are specified.

        :param test_classes:    If left empty, the function loads test documents from all classes
                                (categories). Otherwise, it loads only those classes that are specified.

        :param other_frac:      Which fraction/percentage of unused training files to use for 'other' class.
        """
        if len(train_classes) == 0:
            self.train_classes = reuters.categories()
        else:
            self.train_classes = train_classes[:]

        if len(test_classes) == 0:
            self.test_classes = reuters.categories()
        else:
            self.test_classes = test_classes[:]

        if other_frac is None:
            self.train = [d for d in reuters.fileids() if d.startswith('training/')
                          and len(reuters.categories(d)) == 1 and reuters.categories(d)[0] in self.train_classes]
        else:
            self.train = [d for d in reuters.fileids() if d.startswith('training/') and len(reuters.categories(d)) == 1
                          and (reuters.categories(d)[0] in self.train_classes or random() < other_frac)]
        self.test = [d for d in reuters.fileids() if d.startswith('test/')
                     and len(reuters.categories(d)) == 1 and reuters.categories(d)[0] in self.test_classes]

        # A numpy 2d array of feature vectors.
        self.data['x_train'] = self.vectorizer.fit_transform([reuters.raw(d) for d in self.train]).toarray()
        if other_frac is None:
            # A numpy array of class labels.
            self.data['y_train'] = self.labeler.fit_transform([reuters.categories(d)[0] for d in self.train])
        else:
            self.data['y_train'] = self.labeler.fit_transform(
                [reuters.categories(d)[0] if reuters.categories(d)[0] in self.train_classes else 'other' for d in self.train]
            )
        # A numpy 2d array of feature vectors.
        self.data['x_test'] = self.vectorizer.transform([reuters.raw(d) for d in self.test]).toarray()
        # A numpy array that contains 1s for true novelties and 0s for non-novelties.
        self.data['y_test'] = array([0 if reuters.categories(d)[0] in self.train_classes else 1 for d in self.test])

        if len(self.train_classes) == 2:
            self.data['y_train'] = self.data['y_train'].ravel()
        # if len(self.test_classes) == 2:
        #     self.data['y_test'] = self.data['y_test'].ravel()

    def summary(self):
        """
        Prints the contents of the dataset: file ids and their corresponding categories.
        """
        print('Train')
        print('_' * 42)
        print('{:>20}'.format('Document') + ' |' + '{:>20}'.format('Category'))
        print('-' * 42)
        for i in range(len(self.train)):
            print('{:>20}'.format(self.train[i]) + ' |' + '{:>20}'.format(reuters.categories(self.train[i])[0]))
        print('_' * 42)
        print('Test')
        print('_' * 42)
        print('{:>20}'.format('Document') + ' |' + '{:>20}'.format('Category'))
        print('-' * 42)
        for i in range(len(self.test)):
            print('{:>20}'.format(self.test[i]) + ' |' + '{:>20}'.format(reuters.categories(self.test[i])[0]))

    def stats(self):
        """
        :return:    Important statistics about the dataset - numbers of documents in different classes with
                    corresponding percentages, as well as vocabulary sizes for every class.
        """
        lt = LemmaTokenizer()
        train_stats = {}
        test_stats = {}

        for c in reuters.categories():
            train_stats[c] = {'num_of_docs': 0, 'percentage': 0.0, 'words': set([])}
            test_stats[c] = {'num_of_docs': 0, 'percentage': 0.0, 'words': set([])}

        for d in self.train:
            c = reuters.categories(d)[0]
            train_stats[c]['num_of_docs'] += 1
            train_stats[c]['words'] |= set(lt.lemma_tokenize(reuters.raw(d)))
        for d in self.test:
            c = reuters.categories(d)[0]
            test_stats[c]['num_of_docs'] += 1
            test_stats[c]['words'] |= set(lt.lemma_tokenize(reuters.raw(d)))

        s_train = sum(train_stats[c]['num_of_docs'] for c in train_stats.keys())
        s_test = sum(test_stats[c]['num_of_docs'] for c in test_stats.keys())

        res = ({}, {})

        for c in train_stats.keys():
            if train_stats[c]['num_of_docs'] != 0:
                train_stats[c]['percentage'] = train_stats[c]['num_of_docs'] / s_train
                train_stats[c]['words'] = len(train_stats[c]['words'])
                res[0][c] = train_stats[c]
        for c in test_stats.keys():
            if test_stats[c]['num_of_docs'] != 0:
                test_stats[c]['percentage'] = test_stats[c]['num_of_docs'] / s_test
                test_stats[c]['words'] = len(test_stats[c]['words'])
                res[1][c] = test_stats[c]

        return res

class DirLoader:
    """
    Used to load an arbitrary dataset.
    """
    def __init__(self, vectorizer, train_path, test_path=None):
        """
        :param vectorizer:  Used to convert raw text documents (strings) into vectors of numbers. It
                            can be one of vectorizers available in sklearn, or a custom vectorizer, as
                            long as it provides fit_transform and transform functions in the same
                            format as they are defined in sklearn.

        :param train_path:  Path to directory that contains the training set. This directory needs to
                            contain sub-directories corresponding to different categories of documents.

        :param test_path:   Path to directory that contains the test set. If not None, this directory
                            needs to contain one sub-directory for documents that belong to one of the
                            initial categories and one sub-directory for true novelties (documents that
                            belong to a category never seen before). Folder names should be in this order
                            (for example: '1' for non-novelties and '2' for novelties, or 'non-novelty'
                            and 'novelty' etc).
        """
        self.train_path = train_path
        self.test_path = test_path
        # A list of class names (categories to which a document may belong).
        self.classes = []
        # A list of raw texts in the training set.
        self.train = []
        # A list of raw texts in the test set.
        self.test = []
        # A dictionary that contains the entire dataset, converted into a numeric representation.
        self.data = {'x_train': [], 'y_train': [], 'x_test': [], 'y_test': []}
        self.vectorizer = vectorizer

    def load_data(self):
        """
        Build the self.data dictionary.
        """
        temp_train = load_files(self.train_path, encoding='latin1')
        self.classes = temp_train['target_names']
        self.train = temp_train['data']
        self.data['x_train'] = self.vectorizer.fit_transform(self.train).toarray()
        self.data['y_train'] = temp_train['target']
        if len(self.classes) == 2:
            self.data['y_train'] = self.data['y_train'].ravel()
        if self.test_path is not None:
            temp_test = load_files(self.test_path)
            self.test = temp_test['data']
            self.data['x_test'] = self.vectorizer.transform(self.test).toarray()
            self.data['y_test'] = temp_test['target']
