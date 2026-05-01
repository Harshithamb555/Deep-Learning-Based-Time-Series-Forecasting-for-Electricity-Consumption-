# RNN.py
import os
import tensorflow as tf
from keras.models import load_model

class RNN(object):
    def __init__(self):
        self.best_model = None

    def restore(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        # load without custom objects (we didn't use custom metrics in training scripts)
        self.best_model = tf.keras.models.load_model(filepath, compile=False)
        return self.best_model
