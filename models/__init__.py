import time

from keras import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Optimizer
from mxnet.metric import Loss


class BaseClassifier(object):

    __model: Model
    __loss: Loss
    __optimizer: Optimizer

    def __init__(self, backbone, loss, optimizer):
        # Implement model
        self.__model = backbone
        self.__loss = categorical_crossentropy
        self.__optimizer = optimizer

        self.__model.compile(
            optimizer=optimizer,
            loss=self.__loss,
            metrics=['accuracy']
        )

    def fit(self, x, y, hyperparameters):
        initial_time = time.time()

        self.__model.fit(x, y,
                         batch_size=hyperparameters['batch_size'],
                         epochs=hyperparameters['epochs'],
                         callbacks=hyperparameters['callbacks'],
                         validation_data=hyperparameters['val_data']
                         )

        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        self.__model.summary()
        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters['epochs'], eta, time_unit))

    def evaluate(self, test_X, test_Y):
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, x):
        return self.__model.predict(x)

    def save_model(self, file_path):
        self.__model.save(file_path)

    def load_model(self, file_path):
        self.__model.load_model(file_path)

    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model
