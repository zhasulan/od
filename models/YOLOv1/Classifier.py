import keras
from keras_applications.resnet_common import ResNet152, ResNet101

from models import BaseClassifier


class Classifier(BaseClassifier):

    def __init__(self, number_of_classes):
        self.__loss = 'binary_crossentropy'
        self.__optimizer = 'adam'

        self.__model = ResNet152(
            include_top=True
            , weights=None #'imagenet'
            , input_tensor=None
            , input_shape=None
            , pooling=None
            , classes=number_of_classes

            , backend=keras.backend
            , layers=keras.layers
            , models=keras.models
            , utils=keras.utils
        )
        super(Classifier, self).__init__(self.__model, self.__loss, self.__optimizer)


if __name__ == '__main__':
    clf = Classifier(1000)
    print(clf.get_model().summary())
