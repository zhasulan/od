from nets import Backbone


class ResNetBackbone(Backbone):
    """
    ResNet backbone utility function
    """

    def __init__(self, backbone, classes):

        if backbone.find('NASNet') > -1:
            depth = backbone.replace('NASNet', '')

            if depth == 'Large':
                from keras_applications.nasnet import NASNetLarge as NN
            elif depth == 'Mobile':
                from keras_applications.nasnet import NASNetMobile as NN
            else:
                print('Not a valid net')
                raise ValueError

        elif backbone.find('DenseNet') > -1:
            depth = int(backbone.replace('DenseNet', ''))

            if depth == 121:
                from keras_applications.densenet import DenseNet121 as NN
            elif depth == 169:
                from keras_applications.densenet import DenseNet169 as NN
            elif depth == 201:
                from keras_applications.densenet import DenseNet201 as NN
            else:
                print('Not a valid net')
                raise ValueError

        elif backbone.find('MobileNetV2') > -1:
            from keras_applications.mobilenet_v2 import MobileNetV2 as NN
        elif backbone.find('MobileNet') > -1:
            from keras_applications.mobilenet import MobileNet as NN
        elif backbone.find('Inception') > -1:
            depth = backbone.replace('inception', '')

            if depth == 'V3':
                from keras_applications.inception_v3 import InceptionV3 as NN
            elif depth == 'resnetV2':
                from keras_applications.inception_resnet_v2 import InceptionResNetV2 as NN
            else:
                print('Not a valid net')
                raise ValueError

        elif backbone.find('VGG') > -1:
            depth = int(backbone.replace('resnext', ''))

            if depth == 16:
                from keras_applications.vgg16 import VGG16 as NN
            elif depth == 19:
                from keras_applications.vgg19 import VGG19 as NN
            else:
                print('Not a valid net')
                raise ValueError

        elif backbone.find('xception') > -1:
            from keras_applications.xception import Xception as NN
        elif backbone.find('resnext') > -1:
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

            if depth == '50':
                from keras_applications.resnet_common import ResNet50 as NN
            elif depth == '101':
                from keras_applications.resnet_common import ResNet101 as NN
            elif depth == '152':
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

        super(ResNetBackbone, self).__init__(NN, classes)


if __name__ == '__main__':
    clf = ResNetBackbone('resnext101', 1000)
    print(clf.get_model().summary())
