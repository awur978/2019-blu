from keras import backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.initializers import RandomUniform
from keras.regularizers import l2

class APLU(Layer):
    def __init__(self,
                 segments=2,
                 alpha_initializer=RandomUniform(0., 1.),
                 beta_initializer=RandomUniform(0., 1.),
                 alpha_regularizer=l2(1e-3),
                 beta_regularizer=l2(1e-3),
                 shared_axes=None,
                 **kwargs):
        super(APLU, self).__init__(**kwargs)
        self.segments = segments
        self.alpha_initializer = alpha_initializer
        self.beta_initializer = beta_initializer
        self.alpha_regularizer = alpha_regularizer
        self.beta_regularizer = beta_regularizer
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        super(APLU, self).build(input_shape)

        param_shape = [self.segments] + list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i] = 1

        self.alpha = self.add_weight(
            name='alpha',
            shape=param_shape,
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            trainable=True)

        self.beta = self.add_weight(
            name='beta',
            shape=param_shape,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            trainable=True)

    def call(self, x):
        y = K.relu(x)
        for i in range(self.segments):
            y += self.alpha[i] * K.relu(-x + self.beta[i])
        return y
