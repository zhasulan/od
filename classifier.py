import argparse

from models import YOLOv1, FasterRCNN, BaseClassifier
from nets import Backbone, ResNet


def main(args):
    X = None
    Y = None

    backbone = Backbone()

    if args.backbone.find('ResNet') > -1:
        backbone = ResNet(args.number_of_classes)

    clf = BaseClassifier(backbone, None, None)

    if args.method == 'YOLOv1':
        clf = YOLOv1.Classifier(backbone)

    elif args.method == 'FasterRCNN':
        clf = FasterRCNN.Classifier(backbone)

    clf.fit(X, Y)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('method', type=str, choices=['YOLOv1', 'FasterRCNN'],
                        help='Choice one of methods for object detection')

    parser.add_argument('backbone', type=str)
