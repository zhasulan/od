import keras
from keras import Model


class Backbone(object):

    def __init__(self, NN, classes):

        self.__model = NN(
            include_top=True
            , weights='imagenet'
            , input_tensor=None
            , input_shape=None # (224, 224, 3)
            , pooling=None
            , classes=classes

            , backend=keras.backend
            , layers=keras.layers
            , models=keras.models
            , utils=keras.utils
        )

    def get_model(self):
        return self.__model
