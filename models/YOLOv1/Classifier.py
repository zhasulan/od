from keras import Model
from keras.activations import linear
from keras.layers import Dense, Flatten, GlobalAveragePooling2D

from models import BaseClassifier
from nets.ResNet import ResNetBackbone


class Classifier(BaseClassifier):

    def __init__(self, backbone, classes):
        self.__loss = 'binary_crossentropy'
        self.__optimizer = 'adam'

        backbone = ResNetBackbone(backbone, classes)

        super(Classifier, self).__init__(backbone.get_model(), self.__loss, self.__optimizer)

    def train(self, classes):
        x = self.get_model().output

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='probs')(x)
        model = Model(self.get_model().input, x)

        self.set_model(model)


if __name__ == '__main__':
    clf = Classifier('resnet50', 1000)
    clf.train(1000)
    print(clf.get_model().summary())
