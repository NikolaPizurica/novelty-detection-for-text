"""
:author: Nikola Pizurica

Generating novelty detection reports and journals.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ReportGenerator:
    """
    Generates a file that contains information about the novelty detection method employed, as
    well as how text documents are represented in numerical form.
    """
    def __init__(self, vectorizer, detector, y, y_est):
        """
        :param vectorizer:  Vectorizer that was used to represent text documents as vectors of numbers.

        :param detector:    Novelty detector that was used for the task.

        :param y:           Array of true target values (0 for non-novelty, 1 for novelty).

        :param y_est:       Array of estimated target values (0 for non-novelty, 1 for novelty).
        """
        self.vectorizer = vectorizer
        self.detector = detector
        self.y = y
        self.y_est = y_est

    def generate(self, filename):
        """
        :param filename:    Full name of the report file.

        A note on confusion matrix. Non-novelties have the label 0, and novelties have the label 1.
        So, the matrix will have the form:

        [[true_negatives,   false_positives]
         [false_negatives,  true_positives]]
        """
        with open(filename, 'w') as f:
            f.write('Representation:\n')
            f.write(str(self.vectorizer) + '\n')
            f.write('_'*50 + '\n')
            f.write(str(self.detector) + '\n')
            f.write('_'*50 + '\n')
            f.write('Accuracy:\t{}\n'.format(accuracy_score(self.y, self.y_est)))
            f.write('Precision:\t{}\n'.format(precision_score(self.y, self.y_est)))
            f.write('Recall:\t\t{}\n'.format(recall_score(self.y, self.y_est)))
            f.write('F score:\t{}\n'.format(f1_score(self.y, self.y_est)))
            f.write('_' * 50 + '\n')
            f.write('Confusion matrix:\n')
            f.write(str(confusion_matrix(self.y ,self.y_est)))


class LogWriter:
    """
    Generates a file that lists the names of all the text documents that were processed, along with
    their categories and whether a particular file was classified as a novelty.
    """
    def __init__(self, documents, classes, y_est):
        """
        :param documents:   Names of the documents that were processed.

        :param classes:     A list whose elements are class names of corresponding text documents.

        :param y_est:       Array of estimated target values (0 for non-novelty, 1 for novelty).
        """
        self.documents = documents
        self.classes = classes
        self.y_est = y_est

    def write(self, filename):
        """
        :param filename:    Full name of the journal file.
        """
        with open(filename, 'w') as f:
            for i in range(len(self.documents)):
                f.write('Document: ' + str(self.documents[i]) + '\n')
                f.write('Category: ' + str(self.classes[i]) + '\n')
                if self.y_est[i]:
                    f.write('***Novelty***\n')
                f.write('_' * 25 + '\n')
