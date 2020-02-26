"""
:author: Nikola Pizurica

Illustrating the usage of this library.
"""

from src.dataset_loading import ReutersLoader, DirLoader
from src.novelty_detection import MaxConfidenceDetector, ConfidenceDistanceDetector, SVMBasedDetector
from src.visualization import PerformancePlotter, ROCPlotter
from src.preprocessing import LemmaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from numpy import linspace

lt = LemmaTokenizer()

vct = TfidfVectorizer(stop_words=lt.normalize_stopwords(stopwords.words("english")), tokenizer=lt, max_features=500)

r = ReutersLoader(vct)
r.load_data(classes=['earn', 'acq', 'money-fx', 'grain', 'crude'])

nd = ConfidenceDistanceDetector(MultinomialNB())

nd.fit(r.data['x_train'], r.data['y_train'])
predicted = nd.predict(r.data['x_test'], use_heuristic=True)

print('Accuracy:\t{}'.format(accuracy_score(r.data['y_test'], predicted)))
print('Precision:\t{}'.format(precision_score(r.data['y_test'], predicted)))
print('Recall:\t\t{}'.format(recall_score(r.data['y_test'], predicted)))
print('F score:\t{}'.format(f1_score(r.data['y_test'], predicted)))
print(confusion_matrix(r.data['y_test'], predicted))

eps = nd.epsilon
delta = min(eps, 1-eps)

pp = PerformancePlotter(r.data, nd)
pp.performance_curves(linspace(eps-delta, eps+delta, 15), title='Confidence-distance with multinomial naive Bayes')

rp = ROCPlotter(r.data, [MaxConfidenceDetector(LogisticRegression(solver='lbfgs', multi_class='auto')),
                         MaxConfidenceDetector(MultinomialNB()),
                         MaxConfidenceDetector(MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000, alpha=1)),
                         SVMBasedDetector(OneClassSVM(gamma='scale'), nu=0.2)])
rp.roc_curves(['LR max.conf.', 'NB max.conf.', 'NN max.conf.', 'OneClassSVM'], ['b', 'g', 'r', 'y'],
              title='Receiver Operating Characteristic (ROC) curves')
