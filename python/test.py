from __future__ import print_function
from activations import APLU
from keras import backend as K
import numpy as np

x = np.arange(0, 1, 0.01)
y = K.get_value(APLU(segments=5)(K.constant(x)))
for x, y in zip(x, y):
    print(x, y, sep='\t')
