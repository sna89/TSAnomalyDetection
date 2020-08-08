from Models.model import Model
import tensorflow as tf
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
tf.random.set_seed(1)


class LSTM_AE(Model):
    def __init__(self, data):
        super(LSTM_AE, self).__init__(data)

    def run(self):
        pass