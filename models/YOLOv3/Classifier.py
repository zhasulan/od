from tensorflow.python.keras import models, optimizers, losses, activations

from models import BaseClassifier
from models.YOLOv3.darknet53 import darknet


class Classifier(BaseClassifier):

    def __init__(self):
        self.__loss = None
        self.__optimizer = 'adam'
        self.__model = darknet()
        super(Classifier, self).__init__(self.__model, self.__loss, self.__optimizer)


if __name__ == '__main__':
    clf = Classifier()
    print(clf.get_model().summary())
