import json
import keras
import tensorflow as tf
from keras.layers import Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed


class ModelFactory:
    def __init__(self, model_name: str, output_name: str, dropout_rate: float, num_conv_filters: int, use_stride: bool,
                 dataset: str, num_classes: int = 4):

        try:
            if dataset == "BCI2000":
                input_size = json.load(open("./data/args_bci2000_preprocess.txt", 'r'))['grid_points']
            elif dataset == "BCI2aIV":
                input_size = json.load(open("./data/args_bci2aiv_preprocess.txt", 'r'))['grid_points']
            else:
                raise ValueError()
        except FileNotFoundError:
            raise FileNotFoundError("Run preprocessing script first.")

        self.model_name = model_name
        self.output_name = output_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_conv_filters = num_conv_filters
        self.use_stride = use_stride
        self.input_shape = (input_size, input_size, 3)
        self.log_dir = ""

    def get_model(self) -> keras.models.Sequential:
        """
        Returns a new instance of a tensorflow model. The type of model it returns is determined by the parameter set
        on initializing the model factory.

        Returns:
            Compiled keras tensorflow model
        """

        if self.model_name == 'eegA':
            model = self.get_eegA_model()
        elif self.model_name == 'eegB':
            model = self.get_eegB_model()
        elif self.model_name == 'eegC':
            model = self.get_eegC_model()
        elif self.model_name == 'eegD':
            model = self.get_eegD_model()
        elif self.model_name == 'eegA_LSTM':
            model = self.get_eegA_LSTM_model()
        elif self.model_name == 'eegD_LSTM':
            model = self.get_eegD_LSTM_model()
        else:
            raise ValueError

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        return model

    def get_eegA_model(self):
        model = Sequential()

        first_block_strides = (2, 2) if self.use_stride else (1, 1)

        model.add(Conv2D(input_shape=self.input_shape, filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters, strides=first_block_strides, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=self.num_classes, activation='softmax'))
        return model

    def get_eegB_model(self):
        model = Sequential()

        first_block_strides = (2, 2) if self.use_stride else (1, 1)

        model.add(Conv2D(input_shape=self.input_shape, filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters, strides=first_block_strides, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=self.num_classes, activation='softmax'))
        return model

    def get_eegC_model(self):
        model = Sequential()

        first_block_strides = (2, 2) if self.use_stride else (1, 1)

        model.add(Conv2D(input_shape=self.input_shape, filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters, strides=first_block_strides, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=self.num_conv_filters * 4, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=self.num_classes, activation='softmax'))
        return model

    def get_eegD_model(self):
        model = Sequential()

        first_block_strides = (2, 2) if self.use_stride else (1, 1)

        model.add(Conv2D(input_shape=self.input_shape, filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters, strides=first_block_strides, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=self.num_conv_filters * 4, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=self.num_classes, activation='softmax'))
        return model

    def get_eegA_LSTM_model(self):
        model = Sequential()

        first_block_strides = (2, 2) if self.use_stride else (1, 1)
        
        model.add(TimeDistributed(Conv2D(input_shape=self.input_shape, filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(Conv2D(filters=self.num_conv_filters, strides=first_block_strides, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(128))

        model.add(Dropout(rate=0.5))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=self.num_classes, activation='softmax'))
        return model

    def get_eegD_LSTM_model(self):
        model = Sequential()

        first_block_strides = (2, 2) if self.use_stride else (1, 1)

        model.add(TimeDistributed(Conv2D(input_shape=self.input_shape, filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(Conv2D(filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(Conv2D(filters=self.num_conv_filters, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(Conv2D(filters=self.num_conv_filters, strides=first_block_strides, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(Conv2D(filters=self.num_conv_filters * 2, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(filters=self.num_conv_filters * 4, kernel_size=(3, 3), padding="same", activation="relu")))
        model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(128))

        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(units=self.num_classes, activation='softmax'))
        return model

    def get_callbacks(self, patience: int, log_dir_suffix: str):
        """
        Args:
            patience: patience parameter of the 'EarlyStopping' callback
            log_dir_suffix: descriptive string used to append at the end of the log directory

        Returns:
            A list of callbacks for fitting a model
        """
        self.log_dir = f"logs/fit/{self.output_name}{log_dir_suffix}"
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1),
                     tf.keras.callbacks.LearningRateScheduler(lambda e: 0.001 if e < 20 else 0.0004 if e < 40 else 0.0001),
                     tf.keras.callbacks.EarlyStopping(monitor='accuracy',
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
