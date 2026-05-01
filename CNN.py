# CNN.py
import os
import tensorflow as tf

class CNN(object):
    def __init__(self):
        self.best_model = None

    def restore(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        self.best_model = tf.keras.models.load_model(filepath, compile=False)
        return self.best_model
