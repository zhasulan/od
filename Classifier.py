from keras import Model
from keras.engine.saving import load_model
from keras.optimizers import Optimizer
from mxnet.metric import Loss


class Classifier(object):
    __model: Model
    __loss: Loss
    __optimizer: Optimizer
    __metric: None

    def __init__(self, backbone, loss, optimizer, metric):
        self.__model = backbone
        self.__loss = loss
        self.__optimizer = optimizer
        self.__metric = metric

        self.__model.compile(
            optimizer=self.__optimizer,
            loss=self.__loss,
            metrics=self.__metric
        )

    def last_layers(self, add_function, classes):
        self.__model = Model(self.__model.input, add_function(self.__model.output, classes))

    def fit(self, x, y):
        self.__model.fit(x, y)

    def evaluate(self, x, y):
        return self.__model.evaluate(x, y)

    def predict(self, x):
        return self.__model.predict(x)

    def save_model(self, file_path):
        self.__model.save(file_path)

    def load_model(self, file_path):
        self.__model = load_model(file_path)
