from collections import namedtuple
Classifier = namedtuple('Classifier', 'classifier idx_to_class')


class FaceRecogn:
    def __init__(self, classifier, idx_to_class):
        self.classifier = classifier
        self.idx_to_class = idx_to_class

    def recognise_faces(self):

        return [Classifier(classifier=self.classifier, idx_to_class=self.idx_to_class)]

    def __call__(self):
        return self.recognise_faces()
