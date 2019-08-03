from nets import Backbone


class ResNetBackbone(Backbone):
    """
    ResNet backbone utility function
    """

    def __init__(self, backbone, number_of_classes):

        if backbone.find('resnext') > -1:
            depth = int(backbone.replace('resnext', ''))

            if depth == 50:
                from keras_applications.resnet_common import ResNeXt50 as NN
            elif depth == 101:
                from keras_applications.resnet_common import ResNeXt101 as NN
            else:
                print('Not a valid net')
                raise ValueError

        elif backbone.find('resnet') > -1:
            depth = backbone.replace('resnet', '')

            if depth == 50:
                from keras_applications.resnet_common import ResNet50 as NN
            elif depth == 101:
                from keras_applications.resnet_common import ResNet101 as NN
            elif depth == 152:
                from keras_applications.resnet_common import ResNet152 as NN
            elif depth == '50V2':
                from keras_applications.resnet_common import ResNet50V2 as NN
            elif depth == '101V2':
                from keras_applications.resnet_common import ResNet101V2 as NN
            elif depth == '152V2':
                from keras_applications.resnet_common import ResNet152V2 as NN
            else:
                print('Not a valid net')
                raise ValueError
        else:
            print('Not a valid net')
            raise ValueError

        super(ResNetBackbone, self).__init__(NN, number_of_classes)


if __name__ == '__main__':
    clf = ResNetBackbone('resnext101', 1000)
    print(clf.get_model().summary())
