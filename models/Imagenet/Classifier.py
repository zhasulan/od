import keras
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

from models import BaseClassifier
from nets.ResNet import ResNetBackbone


class Classifier(BaseClassifier):

    def __init__(self, backbone, number_of_classes):
        self.__loss = 'binary_crossentropy'
        self.__optimizer = 'adam'

        backbone = ResNetBackbone(backbone, number_of_classes)

        super(Classifier, self).__init__(backbone.get_model(), self.__loss, self.__optimizer)

    def train(self):
        # super(Classifier, self).train()
        None


if __name__ == '__main__':
    clf = Classifier('resnet101V2', 1000)

    filename = '../../images/cat.jpg'
    # load an image in PIL format
    original = load_img(filename, target_size=(224, 224))
    print('PIL image size', original.size)
    plt.imshow(original)
    plt.show()

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)
    plt.imshow(np.uint8(numpy_image))
    plt.show()
    print('numpy array size', numpy_image.shape)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('image batch size', image_batch.shape)
    plt.imshow(np.uint8(image_batch[0]))

    processed_image = preprocess_input(image_batch)

    # get the predicted probabilities for each class
    predictions = clf.predict(processed_image)
    # print predictions

    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label = decode_predictions(predictions)[0][0]
    print(label)

    # print(clf.get_model().summary())
