from keras import backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.initializers import RandomUniform

class ScaledReLU(Layer):
    def __init__(self,
                 alpha=0.5,
                 beta=1.5,
                 parametric_alpha=False,
                 parametric_beta=False,
                 alpha_initializer='zeros',
                 beta_initializer='ones',
                 alpha_regularizer=None,
                 beta_regularizer=None,
                 shared_axes=None,
                 **kwargs):
        super(ScaledReLU, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.parametric_alpha = parametric_alpha
        self.parametric_beta = parametric_beta
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
        super(ScaledReLU, self).build(input_shape)

        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1

        if self.parametric_alpha:
            self.alpha = self.add_weight(
                name='alpha',
                shape=param_shape,
                initializer=self.alpha_initializer,
                regularizer=self.alpha_regularizer,
                trainable=True)

        if self.parametric_beta:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                trainable=True)

        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)

    def call(self, x):
        neg = self.alpha * -K.relu(-x)
        pos = self.beta * K.relu(x)
        return neg + pos

    def compute_output_shape(self, input_shape):
        return input_shape
