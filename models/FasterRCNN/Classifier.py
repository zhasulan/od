from tensorflow.python.keras import models , optimizers , losses ,activations

from models import BaseClassifier


class Classifier(BaseClassifier):

    def __init__(self, backbone):
        self.__loss = None
        super(Classifier, self).__init__(backbone, self.__loss, )
