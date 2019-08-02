import keras

from nets import Backbone


class ResNetBackbone(Backbone):
    """
    ResNet backbone utility function
    """

    def ___init__(self, number_of_classes):
        super(ResNetBackbone, self).__init__()

        depth = int(self.backbone.replace('resnet', ''))

        if depth == 50:
            from keras_applications.resnet_common import ResNet50 as NN
        elif depth == 101:
            from keras_applications.resnet_common import ResNet101 as NN
        else:
            print('Not a valid net')
            raise ValueError

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
