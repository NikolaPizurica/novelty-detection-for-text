"""
:author: Nikola Pizurica

Illustrating the usage of this library.
"""

from src.dataset_loading import ReutersLoader, DirLoader
from src.novelty_detection import MaxConfidenceDetector, ConfidenceDistanceDetector, SVMBasedDetector, ExtendedMaxConfidenceDetector
from src.visualization import PerformancePlotter, ROCPlotter, ComparisonPlotter
from src.preprocessing import LemmaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from numpy import linspace
from nltk.corpus import reuters


# Loading a specific setting of the Reuters dataset

lt = LemmaTokenizer()
vct = TfidfVectorizer(stop_words=lt.normalize_stopwords(stopwords.words("english")), tokenizer=lt, max_features=500)
r = ReutersLoader(vct)
r.load_data(train_classes=['earn', 'acq', 'crude', 'trade', 'money-fx'])
r.summary()
print()


# Testing the performance of one novelty detector

nd = MaxConfidenceDetector(LogisticRegression(solver='lbfgs', multi_class='auto'), epsilon=0.8)
nd.fit(r.data['x_train'], r.data['y_train'])
predicted = nd.predict(r.data['x_test'])
print('Accuracy:\t{}'.format(accuracy_score(r.data['y_test'], predicted)))
print('Precision:\t{}'.format(precision_score(r.data['y_test'], predicted)))
print('Recall:\t\t{}'.format(recall_score(r.data['y_test'], predicted)))
print('F score:\t{}'.format(f1_score(r.data['y_test'], predicted)))
print(confusion_matrix(r.data['y_test'], predicted))
print()


# Plotting ROC curves of various novelty detectors.

rp = ROCPlotter(
    r.data,
    [
        MaxConfidenceDetector(LogisticRegression(solver='lbfgs', multi_class='auto')),
        MaxConfidenceDetector(MultinomialNB()),
        MaxConfidenceDetector(MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000, alpha=1)),
        SVMBasedDetector(OneClassSVM(gamma='scale'), nu=0.2)
    ]
)
rp.roc_curves(
    ['LR max.conf.', 'NB max.conf.', 'NN max.conf.', 'OneClassSVM'],
    ['b', 'g', 'r', 'y',],
    title='Receiver Operating Characteristic (ROC) curves'
)
print()


# Comparing important performance measures for different novelty detectors, on multiple datasets.
# Be aware that this generally takes a long time to run.

cp = ComparisonPlotter(
    r,
    [
        MaxConfidenceDetector(LogisticRegression(solver='lbfgs', multi_class='auto'), epsilon=0.9),
        MaxConfidenceDetector(MultinomialNB(), epsilon=0.9),
        MaxConfidenceDetector(MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000, alpha=1), epsilon=0.9)
    ]
)
cp.performance_curves(
    [accuracy_score, recall_score, precision_score],
    [(['earn', 'acq'], []), (['earn', 'acq', 'crude', 'trade', 'money-fx'], [])],
    ['LR max.conf.', 'NB max.conf.', 'NN max.conf.'],
    ['b', 'g', 'r'],
    ['Accuracy', 'Recall', 'Precision'],
    [r'Trained on $D^{(2)}_{train}$ and tested on $D_{test}$', r'Trained on $D^{(5)}_{train}$ and tested on $D_{test}$']
)
print()


# Class-level leave-one-out cross validation. This is a time-consuming process as well.

cp.leave_one_out(
    ['earn', 'acq', 'crude', 'trade', 'money-fx'],
    ['LR max.conf.', 'NB max.conf.', 'NN max.conf.'],
    ['b', 'g', 'r']
)
print()
