from keras import backend as K
from keras.constraints import Constraint

class WeightClip(Constraint):
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, p):
        return K.clip(p, self.min, self.max)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'min': self.min,
                'max': self.max}
