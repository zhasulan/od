import argparse

from models import YOLOv1, FasterRCNN, BaseClassifier
from nets import Backbone, ResNet


def main(args):
    X = None
    Y = None

    backbone = Backbone()

    if args.backbone.find('ResNet') > -1:
        backbone = ResNet(args.number_of_classes)

    if args.model == 'YOLOv1':
        from models.YOLOv1 import Classifier as Classifier 
    elif args.model == 'FasterRCNN':
        from models.FasterRCNN import Classifier as Classifier
    else:
        print('Not a valid model')
        raise ValueError
        
    clf = Classifier(backbone)
    clf.fit(X, Y)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, choices=['YOLOv1', 'FasterRCNN'],
                        help='Choice one of methods for object detection')

    parser.add_argument('backbone', type=str)
