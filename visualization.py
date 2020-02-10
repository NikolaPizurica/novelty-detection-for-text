"""
:author: Nikola Pizurica

Various visualization utilities.
"""

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import array


class DatasetVisualizer:
    """
    Plotting the dataset by reducing its dimension to 2. t-SNE dimensionality reduction is used and
    it can be slow for large datasets. Different colors correspond to different classes.
    """
    def __init__(self, X, y):
        """
        :param X:       Matrix (numpy 2d array) of n input vectors with m features.

        :param y:       Array of n target values.
        """
        self.X = X
        self.y = y
        self.reduced = TSNE(n_components=2, random_state=42).fit_transform(self.X)

    def plot(self, labels, colors, save=None):
        """
        :param labels:  Only instances with specified labels will be plotted.

        :param colors:  Which colors to use for different classes.

        :param save:    If a string is passed, the plot is automatically saved to a file with this name.
                        Otherwise (if left None) the plt.show() function is called.
        """
        plt.clf()
        temp_X = array([[self.reduced[i, 0], self.reduced[i, 1]] for i in range(len(self.y)) if self.y[i] in labels])
        temp_y = array([self.y[i] for i in range(len(self.y)) if self.y[i] in labels])
        plt.scatter(temp_X[:, 0], temp_X[:, 1], c=temp_y, cmap=ListedColormap(colors))
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()


class PerformancePlotter:
    """
    Graphing various performance metrics as functions of a specified parameter.
    """
    def __init__(self, data, detector):
        """
        :param data:        A dictionary that must contain 'x_train', 'y_train' and 'y_test' keys with
                            respective values (a matrix of feature vectors, a vector of class labels
                            and a binary vector that indicates what true novelties are).

        :param detector:    A novelty detector that conforms to conventions described in novelty_detection module.
        """
        self.data = data
        self.detector = detector

    def performance_curves(self, param_values, param_name='epsilon', fit_once=True, use_latex=True, title='', save=None):
        """
        :param param_values:    List of particular values of the specified parameter for which the performance of
                                the detector is evaluated. A graph is obtained by connecting the dots.

        :param param_name:      Name of the varying parameter.

        :param fit_once:        Whether it is enough to fit the model once, at the beginning. This is true for
                                max-confidence and confidence-distance when varying the epsilon parameter.
                                On the other hand, it is false for OneClassSVM when varying the nu parameter.

        :param use_latex:       Whether to use LaTeX formatting for the name of the parameter.

        :param title:           Title to be displayed.

        :param save:            If a string is passed, the plot is automatically saved to a file with this name.
                                Otherwise (if left None) the plt.show() function is called.
        """
        self.detector.fit(self.data['x_train'], self.data['y_train'])
        plt.clf()
        plt.title(title)
        if use_latex:
            plt.xlabel(r'$\{}$'.format(param_name))
        else:
            plt.xlabel(param_name)
        plt.ylabel('performance')
        plt.ylim(0.0, 1.05)
        plt.grid(b=True)
        # A list of accuracy scores for given values of the relevant parameter.
        A = []
        # A list of precision scores for given values of the relevant parameter.
        P = []
        # A list of recall scores for given values of the relevant parameter.
        R = []
        # A list of f1 scores for given values of the relevant parameter.
        F = []
        for param_value in param_values:
            setattr(self.detector, param_name, param_value)
            if not fit_once:
                self.detector.fit(self.data['x_train'], self.data['y_train'])
            predicted = self.detector.predict(self.data['x_test'])
            A.append(accuracy_score(self.data['y_test'], predicted))
            P.append(precision_score(self.data['y_test'], predicted))
            R.append(recall_score(self.data['y_test'], predicted))
            F.append(f1_score(self.data['y_test'], predicted))
        plt.title(title)
        plt.plot(param_values, A, 'b-')
        plt.plot(param_values, P, 'g-')
        plt.plot(param_values, R, 'r-')
        plt.plot(param_values, F, 'y-')
        plt.gca().legend(('Accuracy', 'Precision', 'Recall', 'F score'))
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()


class ROCPlotter:
    """
    Graphing Receiver Operating Characteristic (ROC) curves for a given collection of novelty detectors.
    """
    def __init__(self, data, detectors):
        """
        :param data:        A dictionary that must contain 'x_train', 'x_test' 'y_train' and 'y_test' keys
                            with respective values (two matrices of feature vectors, a vector of class
                            labels and a binary vector that indicates what true novelties are).

        :param detectors:   A list of novelty detectors to be tested.
        """
        self.data = data
        self.detectors = detectors

    def roc_curves(self, nd_labels, colors, title='', save=None):
        """
        :param nd_labels:   Labels of novelty detectors. These are shown in the plot legend.

        :param colors:      Which colors to use when plotting curves.

        :param title:       Title to be displayed.

        :param save:        If a string is passed, the plot is automatically saved to a file with this name.
                            Otherwise (if left None) the plt.show() function is called.
        """
        plt.clf()
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        print('Model', ' '*8, '| AUC score')
        print('-'*35)
        for i in range(len(self.detectors)):
            self.detectors[i].fit(self.data['x_train'], self.data['y_train'])
            fpr, tpr, _ = roc_curve(self.data['y_test'], self.detectors[i].decision_function(self.data['x_test']), pos_label=1)
            plt.plot(fpr, tpr, c=colors[i], lw=1)
            print('{:<15}| {:<20}'.format(nd_labels[i], auc(fpr, tpr)))
        plt.gca().legend(nd_labels)
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()


class NoveltyPlotter:
    """
    Visualizing the dataset through dimensionality reduction and indicating true novelties/non-novelties,
    as well as estimated novelties/non-novelties, for comparison purposes.
    """
    def __init__(self, X, y, y_est):
        """
        :param X:       Matrix (numpy 2d array) of n input vectors with m features.

        :param y:       Array of n true target values.

        :param y_est:   Array of n estimated target values.
        """
        self.X = X
        self.y = y
        self.y_est = y_est
        self.reduced = TSNE(n_components=2, random_state=42).fit_transform(self.X)

    def plot(self, title='', save=None):
        """
        :param title:       Title to be displayed.

        :param save:        If a string is passed, the plot is automatically saved to a file with this name.
                            Otherwise (if left None) the plt.show() function is called.
        """
        plt.clf()
        fig = plt.figure(figsize=plt.figaspect(.5))
        fig.suptitle(title)
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('True classification')
        p1_normal = []
        p1_novelty = []
        p2_normal = []
        p2_novelty = []
        for i in range(len(self.reduced)):
            if self.y[i] == 0:
                p1_normal.append(self.reduced[i, 0])
                p2_normal.append(self.reduced[i, 1])
            else:
                p1_novelty.append(self.reduced[i, 0])
                p2_novelty.append(self.reduced[i, 1])
        normal = ax.scatter(p1_normal, p2_normal, c='blue')
        novelty = ax.scatter(p1_novelty, p2_novelty, c='red')
        ax.legend((normal, novelty), ('Normal', 'Novelty'))
        ax_est = fig.add_subplot(1, 2, 2)
        ax_est.set_title('Estimates')
        p1_normal = []
        p1_novelty = []
        p2_normal = []
        p2_novelty = []
        for i in range(len(self.reduced)):
            if self.y_est[i] == 0:
                p1_normal.append(self.reduced[i, 0])
                p2_normal.append(self.reduced[i, 1])
            else:
                p1_novelty.append(self.reduced[i, 0])
                p2_novelty.append(self.reduced[i, 1])
        normal = ax_est.scatter(p1_normal, p2_normal, c='blue')
        novelty = ax_est.scatter(p1_novelty, p2_novelty, c='red')
        ax_est.legend((normal, novelty), ('Normal', 'Novelty'))
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()