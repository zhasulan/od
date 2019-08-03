import keras


class Backbone(object):

    def __init__(self, NN, number_of_classes):

        self.__model = NN(
            include_top=True
            , weights='imagenet'
            , input_tensor=None
            , input_shape=None
            , pooling=None
            , classes=number_of_classes

            , backend=keras.backend
            , layers=keras.layers
            , models=keras.models
            , utils=keras.utils
        )

    def get_model(self):
        return self.__model
