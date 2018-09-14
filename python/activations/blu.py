from keras import backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.initializers import RandomUniform
from constraints import WeightClip

class BLU(Layer):
    def __init__(self,
                 alpha=0.5,
                 beta=0.5,
                 parametric_alpha=False,
                 parametric_beta=True,
                 alpha_initializer=RandomUniform(0., 1.),
                 beta_initializer=RandomUniform(0., 1.),
                 alpha_regularizer=None,
                 beta_regularizer=None,
                 shared_axes=None,
                 **kwargs):
        super(BLU, self).__init__(**kwargs)
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
        super(BLU, self).build(input_shape)

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
                constraint=WeightClip(0., 1.),
                trainable=True)

        if self.parametric_beta:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=WeightClip(0., 1.),
                trainable=True)

        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)

    def call(self, x):
        return self.beta * (K.sqrt(x * x + self.alpha * self.alpha + K.epsilon()) - self.alpha) + x

    def compute_output_shape(self, input_shape):
        return input_shape
