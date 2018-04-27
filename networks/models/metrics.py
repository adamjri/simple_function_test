from keras import objectives
from keras import backend as K

import numpy as np

_EPSILON = K.epsilon()

def mean_sq_error(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))
