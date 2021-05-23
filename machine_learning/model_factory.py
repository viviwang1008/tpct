import keras
import tensorflow as tf
from datetime import datetime

from keras.callbacks import EarlyStopping
from keras.layers import Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout
from keras.models import Sequential


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.log_dir = ""

    def get_model(self, learning_rate: float, dropout_rate: float) -> keras.models.Sequential:
        """
        Returns a new instance of a tensorflow model. The type of model it returns is determined by the parameter set
        on initializing the model factory.

        Args:
            learning_rate: learning rate parameter for the
            dropout_rate: dropout layer for the dropout layers

        Returns:
            Compiled keras tensorflow model
        """
        if self.model_name == 'eegD':
            pass  # later add "if self.model_name == 'eegD':"... currently only one model
        model = Sequential()

        model.add(Conv2D(input_shape=(64, 64, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=512, activation="relu"))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=4, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        return model

    def get_callbacks(self, patience: int, log_dir_suffix: str):
        """
        Args:
            patience: patience parameter of the 'EarlyStopping' callback
            log_dir_suffix: descriptive string used to append at the end of the log directory

        Returns:
            A list of callbacks for fitting a model
        """
        self.log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{log_dir_suffix}"
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=5),
                     EarlyStopping(monitor='accuracy',
                                   min_delta=0,
                                   patience=patience,
                                   mode='auto',
                                   baseline=None,
                                   restore_best_weights=False)]
        return callbacks

    def get_log_dir(self):
        """
        Returns last used log directory
        """
        return self.log_dir
