import argparse

from models import YOLOv1, FasterRCNN, BaseClassifier
from nets import Backbone, ResNet


def main(args):

    """
    TASKS
    1. All Networks with Imagenet weights
    2. Some special networks (YOLO) with object detection models and weights
    3. Google Object Detection images with labels
    4.
    x. Train function
    x. Fine-tuning function
    x. Transfer learning from Imagenet function


    STEPS
    1. YOLO
    2. Faster-RCNN
    3. Retina Net
    4. Octave Net with models
    5. MobileNetV2
    6.





    :param args:
    :return:
    """


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, choices=['YOLOv1', 'FasterRCNN'],
                        help='Choice one of methods for object detection')

    parser.add_argument('backbone', type=str)
