"""
:author: Nikola Pizurica

This module contains classes that define novelty detectors based on how 'confident' a classifier
is that a document belongs to a particular class. If the confidence level (defined in different
ways for different approaches) is below a predefined boundary, that indicates a novelty.
"""

from numpy import array


class MaxConfidenceDetector:
    """
    A novelty detector that functions by comparing the maximum probability (pmax) in the output
    distribution of a classifier with a fixed threshold (epsilon). If pmax is less than epsilon
    for a ceartain document, that document is labeled as a novelty.
    """
    def __init__(self, model, epsilon=0.8):
        """
        :param model:   A machine learning model used to perform classification. Models available
                        in sklearn can be used, as well as custom, user defined models, as long as
                        they provide fit(X, y) and predict_proba(X) functions, analogous to the
                        ones in sklearn models.

        :param epsilon: Confidence threshold.
        """
        self.model = model
        self.epsilon = epsilon

    def fit(self, X, y):
        """
        :param X:   Matrix (numpy 2d array) of n input vectors with m features.

        :param y:   Array of n target values.
        """
        self.model.fit(X, y)

    def predict(self, X, use_heuristic=False):
        """
        :param X:               Matrix (numpy 2d array) of n input vectors with m features.

        :param use_heuristic:   Whether to calculate epsilon as an average over max probabilities for the
                                whole test set. If True, previosly specified value of epsilon is overwritten.

        :return:                Binary array of length n. k-th element in this array is 1 if k-th vector
                                in the input represents a novelty and 0 otherwise.
        """
        P = self.model.predict_proba(X)
        max_proba = array([max(P[i]) for i in range(len(P))])
        if use_heuristic:
            self.epsilon = max_proba.mean()
        return array([1 if max_proba[i] < self.epsilon else 0 for i in range(len(max_proba))])

    def decision_function(self, X):
        """
        Mainly a utility for graphing ROC curves.

        :param X:   Matrix of n input vectors with m features.

        :return:    Array of length n.
        """
        P = self.model.predict_proba(X)
        return array([-max(P[i]) for i in range(len(P))])

    def __str__(self):
        """
        :return:    String that contains all information about the detector.
        """
        return 'Model:\n' + str(self.model) + '\n\nConfidence threshold:\np_max < ' + str(self.epsilon)


class ConfidenceDistanceDetector:
    """
    A novelty detector that functions by comparing the absolute difference between two highest
    probabilities in the output distribution of a classifier (delta) with a fixed threshold
    epsilon. If delta is less than epsilon for a certain document, it is labeled as a novelty.
    """
    def __init__(self, model, epsilon=0.8):
        """
        :param model:   A machine learning model used to perform classification. Models available
                        in sklearn can be used, as well as custom, user defined models, as long as
                        they provide fit(X, y) and predict_proba(X) functions, analogous to the
                        ones in sklearn models.

        :param epsilon: Distance threshold.
        """
        self.model = model
        self.epsilon = epsilon

    def fit(self, X, y):
        """
        :param X:   Matrix of n input vectors with m features.

        :param y:   Array of n target values.
        """
        self.model.fit(X, y)

    def predict(self, X, use_heuristic=False):
        """
        :param X:               Matrix of n input vectors with m features.

        :param use_heuristic:   Whether to calculate epsilon as an average over probability distances for the
                                whole test set. If True, previosly specified value of epsilon is overwritten.

        :return:                Binary array of length n. k-th element in this array is 1 if k-th vector
                                in the input represents a novelty and 0 otherwise.
        """
        P = [sorted(row, reverse=True) for row in self.model.predict_proba(X)]
        proba_dist = array([P[i][0] - P[i][1] for i in range(len(P))])
        if use_heuristic:
            self.epsilon = proba_dist.mean()
        return array([1 if proba_dist[i] < self.epsilon else 0 for i in range(len(proba_dist))])

    def decision_function(self, X):
        """
        Mainly a utility for graphing ROC curves.

        :param X:   Matrix of n input vectors with m features.

        :return:    Array of length n.
        """
        P = [sorted(row, reverse=True) for row in self.model.predict_proba(X)]
        return array([-(P[i][0] - P[i][1]) for i in range(len(P))])

    def __str__(self):
        """
        :return:    String that contains all information about the detector.
        """
        return 'Model:\n' + str(self.model) + '\n\nDistance threshold:\np_first - p_second < ' + str(self.epsilon)

class SVMBasedDetector:
    """
    A helper class to test the performance of OneClassSVM more easily.
    """
    def __init__(self, model, nu=0.2):
        self.nu = nu
        self.model = model
    def fit(self, X, y):
        self.model.set_params(nu=self.nu)
        self.model.fit(X)
    def predict(self, X):
        return (1 - self.model.predict(X)) // 2
    def decision_function(self, X):
        return -self.model.decision_function(X)
